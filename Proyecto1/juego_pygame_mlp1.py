import os
import csv
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pygame
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.ion()

BASE_W, BASE_H = 1080, 720
WINDOW_FRACTION = 0.97
EXTRA_SCALE = 1.1

# Tipos de modelo disponibles
TIPO_MLP  = "MLP"
TIPO_ARBOL = "Árbol"


# Etiquetas multiclase para el modelo
ACCION_QUIETO   = 0  # sin movimiento
ACCION_SALTO    = 1  # saltando
ACCION_AGACHADO = 2  # agachado

# Frames que dura el agachado en modo AUTO antes de levantarse solo.
# A 45 FPS: 4 frames ≈ 89 ms — mínimo visible pero repetible muy rápido.
AGACHADO_DURACION_AUTO = 4


@dataclass
class Sample:
    velocidad_bala: float
    distancia: float
    altura_bala: float   # 0=suelo, 1=cintura, 2=cabeza
    salto: int           # 1 si está en el aire, 0 si no
    agachado: int        # 1 si está agachado, 0 si no

    @property
    def accion(self) -> int:
        """Convierte las columnas salto/agachado a etiqueta multiclase."""
        if self.salto == 1:
            return ACCION_SALTO
        if self.agachado == 1:
            return ACCION_AGACHADO
        return ACCION_QUIETO


@dataclass
class EstadisticasBala:
    """Contadores de frames por estado durante el vuelo de una bala."""
    frames_salto: int = 0
    frames_agachado: int = 0
    frames_estatico: int = 0

    def total(self) -> int:
        return self.frames_salto + self.frames_agachado + self.frames_estatico

    def porcentajes(self) -> Tuple[float, float, float]:
        t = self.total()
        if t == 0:
            return 0.0, 0.0, 0.0
        return (
            self.frames_salto    / t * 100,
            self.frames_agachado / t * 100,
            self.frames_estatico / t * 100,
        )


class Juego:
    def __init__(self) -> None:
        pygame.init()

        self._flags = 0
        self._fullscreen = False

        start_w = BASE_W
        start_h = BASE_H
        self.pantalla = pygame.display.set_mode((start_w, start_h), self._flags)
        pygame.display.set_caption("Juego: Bala + salto + agachado + MLP / Árbol")

        self.BLANCO  = (255, 255, 255)
        self.NEGRO   = (0,   0,   0)
        self.GRIS    = (200, 200, 200)
        self.AMARILLO = (255, 220, 120)
        self.VERDE   = (120, 255, 160)
        self.CYAN    = (100, 220, 255)

        self.corriendo = True
        self.modo_auto = False

        # ── Datos ──────────────────────────────────────────────────────────
        self.datos_modelo: List[Sample] = []

        # ── Modelos (MLP y Árbol independientes) ───────────────────────────
        self.modelo_mlp:   Optional[MLPClassifier]      = None
        self.modelo_arbol: Optional[DecisionTreeClassifier] = None
        self.scaler_mlp:   Optional[StandardScaler]     = None
        # El árbol no necesita scaler (es invariante a escala),
        # pero lo guardamos por si cambia en fases futuras.
        self.scaler_arbol: Optional[StandardScaler]     = None

        self.mlp_entrenado   = False
        self.arbol_entrenado = False

        # Clase única (modelo trivial por falta de variedad en los datos)
        self.clase_unica_mlp:   Optional[int] = None
        self.clase_unica_arbol: Optional[int] = None

        # Modelo activo en modo AUTO
        self.tipo_modelo_activo: str = TIPO_ARBOL

        # Debug — ahora guardamos la acción predicha (0/1/2)
        self.ultima_proba_salto: Optional[float] = None
        self.ultima_accion_auto: Optional[int]   = None

        # ── Parámetros de decisión ─────────────────────────────────────────
        self.decision_window        = 500
        self.decision_record_every  = 3
        self._decision_frame_counter = 0

        # ── Geometría / física ─────────────────────────────────────────────
        self.w, self.h = start_w, start_h
        self.scale = 1.0
        self.margin = 50
        self.ground_y = self.h - 100
        self.player_size = (32, 48)
        self.bullet_size = (16, 16)
        self.ship_size   = (64, 64)
        self.fondo_speed = 3

        self.salto = False
        self.en_suelo = True
        self.salto_vel_inicial = 15.0
        self.gravedad = 1.0
        self.salto_vel = self.salto_vel_inicial

        self.agachado = False
        self.agachado_frames_counter = 0
        self.agachado_auto_timer = 0   # frames restantes del agachado automático

        self.current_frame = 0
        self.frame_speed   = 10
        self.frame_count   = 0

        self.velocidad_bala   = -12
        self.bala_disparada   = False
        self.altura_bala_nivel = 0
        self.fondo_x1 = 0
        self.fondo_x2 = start_w

        # ── Estadísticas en tiempo real (HUD) ─────────────────────────────
        self.stats_bala = EstadisticasBala()

        self._apply_resolution(start_w, start_h, reset_positions=True)
        self._reset_estado_juego()

    # ═══════════════════════ resolución / assets ════════════════════════════
    def _apply_resolution(self, w: int, h: int, reset_positions: bool) -> None:
        self.w, self.h = int(w), int(h)

        self.scale = min(self.w / BASE_W, self.h / BASE_H) * EXTRA_SCALE
        self.scale = max(1.0, self.scale)

        self.margin      = int(50 * self.scale)
        ground_offset    = int(100 * self.scale)
        self.ground_y    = self.h - ground_offset

        self.player_size          = (int(32 * self.scale), int(48 * self.scale))
        self.player_size_agachado = (int(32 * self.scale), int(24 * self.scale))
        self.bullet_size          = (int(16 * self.scale), int(16 * self.scale))
        self.ship_size            = (int(64 * self.scale), int(64 * self.scale))
        self.fondo_speed          = max(1, int(2 * self.scale))

        self.salto_vel_inicial = 15 * self.scale
        self.gravedad          = 1  * self.scale
        self.salto_vel         = self.salto_vel_inicial

        self.decision_window = int(500 * self.scale)

        self.fuente       = pygame.font.SysFont("Arial", int(24 * self.scale))
        self.fuente_chica = pygame.font.SysFont("Arial", int(18 * self.scale))

        self._cargar_assets()

        if reset_positions or not hasattr(self, "jugador"):
            self.jugador = pygame.Rect(self.margin, self.ground_y,
                                       self.player_size[0], self.player_size[1])
            self.bala = pygame.Rect(self.w - self.margin,
                                    self.ground_y + int(10 * self.scale),
                                    self.bullet_size[0], self.bullet_size[1])
            self.nave = pygame.Rect(self.w - int(100 * self.scale), self.ground_y,
                                    self.ship_size[0], self.ship_size[1])

    def _cargar_assets(self) -> None:
        def safe_load(path, size, fallback=(200, 200, 200, 255)):
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except Exception:
                surf = pygame.Surface(size, pygame.SRCALPHA)
                surf.fill(fallback)
                return surf

        base = os.path.dirname(__file__)
        self.jugador_frames = [
            safe_load(os.path.join(base, f"assets/sprites/mono_frame_{i}.png"), self.player_size)
            for i in range(1, 5)
        ]
        self.jugador_frames_agachado = [
            safe_load(os.path.join(base, f"assets/sprites/mono_frame_{i}.png"),
                      self.player_size_agachado)
            for i in range(1, 5)
        ]
        self.bala_img  = safe_load(os.path.join(base, "assets/sprites/purple_ball.png"),
                                   self.bullet_size, (160, 120, 255, 255))
        self.fondo_img = safe_load(os.path.join(base, "assets/game/fondo2.png"),
                                   (self.w, self.h), (40, 40, 40, 255))
        self.nave_img  = safe_load(os.path.join(base, "assets/game/ufo.png"),
                                   self.ship_size, (140, 255, 200, 255))

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            info = pygame.display.Info()
            w = info.current_w or self.w
            h = info.current_h or self.h
            self.pantalla = pygame.display.set_mode((w, h), pygame.FULLSCREEN)
            self._apply_resolution(w, h, reset_positions=True)
        else:
            self.pantalla = pygame.display.set_mode((BASE_W, BASE_H), self._flags)
            self._apply_resolution(BASE_W, BASE_H, reset_positions=True)
        self._reset_estado_juego()

    # ═══════════════════════ estado juego / modelos ══════════════════════════
    def _reset_estado_juego(self) -> None:
        self.jugador.x, self.jugador.y = self.margin, self.ground_y
        self.jugador.width, self.jugador.height = self.player_size
        self.nave.x, self.nave.y = self.w - int(100 * self.scale), self.ground_y
        self.bala.x  = self.w - self.margin
        self.bala.y  = self.ground_y + int(10 * self.scale)
        self.bala_disparada    = False
        self.velocidad_bala    = int(-10 * self.scale)
        self.altura_bala_nivel = 0
        self.salto    = False
        self.en_suelo = True
        self.agachado = False
        self.agachado_frames_counter = 0
        self.agachado_auto_timer = 0
        self.salto_vel = self.salto_vel_inicial
        self._decision_frame_counter = 0
        self.fondo_x1 = 0
        self.fondo_x2 = self.w
        # Nota: stats_bala NO se resetea aquí; persiste toda la sesión manual.

    def _reset_modelos(self) -> None:
        """Borra AMBOS modelos."""
        self.modelo_mlp      = None
        self.modelo_arbol    = None
        self.scaler_mlp      = None
        self.scaler_arbol    = None
        self.mlp_entrenado   = False
        self.arbol_entrenado = False
        self.clase_unica_mlp   = None
        self.clase_unica_arbol = None

    def _modelo_activo_entrenado(self) -> bool:
        if self.tipo_modelo_activo == TIPO_MLP:
            return self.mlp_entrenado
        return self.arbol_entrenado

    # ═══════════════════════ export / gráficas ══════════════════════════════
    def exportar_datos_csv(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para exportar."
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")
        try:
            with open(ruta, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["velocidad_bala", "distancia", "altura_bala", "salto", "agachado"])
                for s in self.datos_modelo:
                    writer.writerow([s.velocidad_bala, s.distancia, s.altura_bala,
                                     s.salto, s.agachado])
        except Exception as e:
            return f"Error al guardar CSV: {e}"
        return f"CSV guardado en datos_mlp.csv ({len(self.datos_modelo)} filas)."

    def graficar_datos_2d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia      for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        cs = ["red" if s.salto == 1 else "blue" for s in self.datos_modelo]
        fig_num = plt.figure("Datos MLP - 2D", figsize=(8, 6)).number
        plt.figure(fig_num); plt.clf()
        ax = plt.gca()
        ax.scatter(xs, ys, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia jugador-bala")
        ax.set_ylabel("Velocidad bala")
        ax.set_title("Datos entrenamiento (rojo=salto, azul=no salto)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show(block=False); plt.draw()
        return "Mostrando gráfica 2D."

    def graficar_datos_3d(self) -> str:
        if not self.datos_modelo:
            return "No hay datos para graficar."
        xs = [s.distancia      for s in self.datos_modelo]
        ys = [s.velocidad_bala for s in self.datos_modelo]
        zs = list(range(len(self.datos_modelo)))
        cs = ["red" if s.salto == 1 else "blue" for s in self.datos_modelo]
        fig = plt.figure("Datos MLP - 3D", figsize=(8, 6)); plt.clf()
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(xs, ys, zs, c=cs, alpha=0.6, edgecolors="k", s=30)
        ax.set_xlabel("Distancia"); ax.set_ylabel("Velocidad bala")
        ax.set_zlabel("Índice (tiempo)")
        ax.set_title("Datos entrenamiento 3D (rojo=salto, azul=no salto)")
        plt.tight_layout()
        plt.show(block=False); plt.draw()
        return "Mostrando gráfica 3D."

    # ═══════════════════════ bala / salto / agachado ════════════════════════
    def disparar_bala(self) -> None:
        if not self.bala_disparada:
            self.velocidad_bala    = int(random.randint(-12, -6) * self.scale)
            self.altura_bala_nivel = random.randint(0, 2)
            ph = self.player_size[1]
            if self.altura_bala_nivel == 0:
                bala_y = self.ground_y + int(10 * self.scale)
            elif self.altura_bala_nivel == 1:
                bala_y = self.ground_y - ph // 2
            else:
                bala_y = self.ground_y - ph + int(4 * self.scale)
            self.bala.y       = bala_y
            self.bala_disparada = True

    def reset_bala(self) -> None:
        self.bala.x           = self.w - self.margin
        self.bala.y           = self.ground_y + int(10 * self.scale)
        self.bala_disparada   = False
        self.altura_bala_nivel = 0

    def iniciar_salto(self) -> None:
        if self.en_suelo and not self.agachado:
            self.salto    = True
            self.en_suelo = False

    def manejar_salto(self) -> None:
        if self.salto:
            self.jugador.y -= int(self.salto_vel)
            self.salto_vel -= self.gravedad
            if self.jugador.y >= self.ground_y:
                self.jugador.y    = self.ground_y
                self.salto        = False
                self.salto_vel    = self.salto_vel_inicial
                self.en_suelo     = True

    def iniciar_agachado(self) -> None:
        if self.en_suelo and not self.salto and not self.agachado:
            self.agachado = True
            self.agachado_frames_counter = 0
            self.jugador.height = self.player_size_agachado[1]
            self.jugador.y      = self.ground_y + (self.player_size[1] - self.player_size_agachado[1])

    def dejar_agachado(self) -> None:
        if self.agachado:
            self.agachado       = False
            self.jugador.height = self.player_size[1]
            self.jugador.y      = self.ground_y

    # ═══════════════════════ datos / ML ═════════════════════════════════════
    def registrar_decision_manual(self) -> None:
        if not self.bala_disparada:
            return
        distancia     = abs(self.jugador.x - self.bala.x)
        salto_label   = 0 if self.en_suelo else 1
        agachado_label = 1 if self.agachado else 0
        self.datos_modelo.append(Sample(
            velocidad_bala = float(self.velocidad_bala),
            distancia      = float(distancia),
            altura_bala    = float(self.altura_bala_nivel),
            salto          = salto_label,
            agachado       = agachado_label,
        ))
        # Actualizar estadísticas por bala
        if salto_label == 1:
            self.stats_bala.frames_salto += 1
        elif agachado_label == 1:
            self.stats_bala.frames_agachado += 1
        else:
            self.stats_bala.frames_estatico += 1

    # ── helpers comunes ─────────────────────────────────────────────────────
    def _preparar_datos(self, samples: List[Sample]):
        """Prepara datos para clasificación MULTICLASE: 0=quieto,1=salto,2=agachado."""
        X = [[s.velocidad_bala, s.distancia, s.altura_bala] for s in samples]
        y = [s.accion for s in samples]
        return X, y

    def _clase_unica_msg(self, clase: int, tipo: str) -> str:
        nombres = {ACCION_QUIETO: "SIEMPRE QUIETO",
                   ACCION_SALTO:  "SIEMPRE SALTA",
                   ACCION_AGACHADO: "SIEMPRE AGACHADO"}
        etiqueta = nombres.get(clase, str(clase))
        return (f"[{tipo}] Modelo trivial entrenado ({etiqueta}). "
                "Para más precisión recoge datos de todas las acciones.")

    def cargar_csv(self) -> str:
        """Carga datos desde datos_mlp.csv y los agrega a self.datos_modelo."""
        base = os.path.dirname(__file__)
        ruta = os.path.join(base, "datos_mlp.csv")
        if not os.path.exists(ruta):
            return "No se encontró datos_mlp.csv."
        try:
            cargados = 0
            with open(ruta, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.datos_modelo.append(Sample(
                        velocidad_bala = float(row["velocidad_bala"]),
                        distancia      = float(row["distancia"]),
                        altura_bala    = float(row["altura_bala"]),
                        salto          = int(float(row["salto"])),
                        agachado       = int(float(row["agachado"])),
                    ))
                    cargados += 1
            return f"CSV cargado: {cargados} muestras."
        except Exception as e:
            return f"Error al cargar CSV: {e}"

    # ── MLP ─────────────────────────────────────────────────────────────────
    def entrenar_mlp(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas >= 80 muestras. Juega en MANUAL o carga CSV."
        X, y = self._preparar_datos(samples)
        clases = sorted(set(y))
        if len(clases) < 2:
            self.modelo_mlp    = None
            self.scaler_mlp    = None
            self.clase_unica_mlp = int(clases[0])
            self.mlp_entrenado = True
            return True, self._clase_unica_msg(self.clase_unica_mlp, TIPO_MLP)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                   random_state=42, stratify=y)
        scaler  = StandardScaler()
        X_tr    = scaler.fit_transform(X_tr)
        X_te    = scaler.transform(X_te)
        # class_weight='balanced' compensa desbalanceo entre quieto/salto/agachado
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=0.001,
        )
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        dist = {c: y_tr.count(c) for c in set(y_tr)}
        self.modelo_mlp      = clf
        self.scaler_mlp      = scaler
        self.clase_unica_mlp = None
        self.mlp_entrenado   = True
        return True, f"MLP Accuracy: {acc:.3f}"

    # ── Árbol de decisión ───────────────────────────────────────────────────
    def entrenar_arbol(self) -> Tuple[bool, str]:
        samples = list(self.datos_modelo)
        if len(samples) < 80:
            return False, "Necesitas >= 80 muestras. Juega en MANUAL o carga CSV."
        X, y = self._preparar_datos(samples)
        clases = sorted(set(y))
        if len(clases) < 2:
            self.modelo_arbol      = None
            self.scaler_arbol      = None
            self.clase_unica_arbol = int(clases[0])
            self.arbol_entrenado   = True
            return True, self._clase_unica_msg(self.clase_unica_arbol, TIPO_ARBOL)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                   random_state=42, stratify=y)
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)
        # class_weight='balanced' para que el árbol no ignore el agachado
        clf = DecisionTreeClassifier(max_depth=8, random_state=42,
                                     class_weight="balanced")
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        dist = {c: y_tr.count(c) for c in set(y_tr)}
        self.modelo_arbol      = clf
        self.scaler_arbol      = scaler
        self.clase_unica_arbol = None
        self.arbol_entrenado   = True
        return True, f"Árbol Accuracy: {acc:.3f}"

    # ── Inferencia AUTO ─────────────────────────────────────────────────────
    def _predecir_accion(self, modelo, scaler, clase_unica) -> int:
        """Devuelve la acción predicha: 0=quieto, 1=salto, 2=agachado."""
        if clase_unica is not None and modelo is None:
            return int(clase_unica)
        if modelo is None or scaler is None:
            return ACCION_QUIETO
        distancia = abs(self.jugador.x - self.bala.x)
        X  = [[float(self.velocidad_bala), float(distancia),
               float(self.altura_bala_nivel)]]
        Xs = scaler.transform(X)
        pred = int(modelo.predict(Xs)[0])
        # Guardar probabilidad de salto para HUD (índice 1 si existe)
        if hasattr(modelo, "predict_proba"):
            probas = modelo.predict_proba(Xs)[0]
            clases = list(modelo.classes_)
            idx1 = clases.index(ACCION_SALTO) if ACCION_SALTO in clases else -1
            self.ultima_proba_salto = float(probas[idx1]) if idx1 >= 0 else 0.0
        return pred

    def decision_auto(self) -> int:
        """Devuelve la acción que debe ejecutar el personaje en modo auto.
        0=quieto, 1=salto, 2=agachado.
        Solo actúa cuando hay bala disparada; fuera de eso devuelve quieto.
        """
        if not self._modelo_activo_entrenado():
            return ACCION_QUIETO
        if not self.bala_disparada:
            return ACCION_QUIETO

        if self.tipo_modelo_activo == TIPO_MLP:
            accion = self._predecir_accion(self.modelo_mlp, self.scaler_mlp,
                                           self.clase_unica_mlp)
        else:
            accion = self._predecir_accion(self.modelo_arbol, self.scaler_arbol,
                                           self.clase_unica_arbol)

        self.ultima_accion_auto = accion
        return accion

    # Alias para compatibilidad / HUD
    def decision_auto_saltar(self) -> bool:
        """Wrapper legado — solo indica si la acción es saltar."""
        return self.decision_auto() == ACCION_SALTO

    # ═══════════════════════ menú ═══════════════════════════════════════════
    def _dibujar_menu(self, msg: str = "") -> None:
        self.pantalla.fill(self.NEGRO)
        titulo = self.fuente.render("MENÚ", True, self.BLANCO)
        self.pantalla.blit(titulo,
                           (self.w // 2 - titulo.get_width() // 2, int(60 * self.scale)))

        # Estado de modelos
        mlp_ok   = "✓" if self.mlp_entrenado   else "✗"
        arbol_ok = "✓" if self.arbol_entrenado else "✗"
        activo_label = f"[activo: {self.tipo_modelo_activo}]"

        opciones = [
            "M  - Manual  (reinicia dataset y borra modelos)",
            f"A  - Auto    (usa modelo activo) {activo_label}",
            f"T  - Entrenar ambos modelos  MLP[{mlp_ok}] Árbol[{arbol_ok}]",
            "L  - Alternar modelo activo",
            "C  - Exportar datos a CSV",
            "I  - Importar datos de CSV (datos_mlp.csv)",
            "F  - Fullscreen (toggle)",
            "Q  - Salir",
        ]
        x0     = int(80 * self.scale)
        y      = int(130 * self.scale)
        line_h = self.fuente.get_linesize()
        pad    = max(5, int(5 * self.scale))
        for op in opciones:
            color = self.AMARILLO if "activo" in op else self.BLANCO
            t = self.fuente.render(op, True, color)
            self.pantalla.blit(t, (x0, y))
            y += line_h + pad

        y += int(8 * self.scale)
        estado = [
            f"Memoria: {len(self.datos_modelo)} muestras",
            f"Resolución: {self.w}x{self.h}  scale≈{self.scale:.2f}",
        ]
        for line in estado:
            t = self.fuente_chica.render(line, True, self.GRIS)
            self.pantalla.blit(t, (x0, y))
            y += self.fuente_chica.get_linesize()

        if msg:
            mm = self.fuente_chica.render(msg, True, self.AMARILLO)
            self.pantalla.blit(mm, (x0, y + int(12 * self.scale)))

        pygame.display.flip()

    def mostrar_menu(self) -> None:
        msg = ""
        esperando = True
        self._decision_frame_counter = 0
        while esperando and self.corriendo:
            self._dibujar_menu(msg)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                    esperando = False
                    break
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_m:
                        self.modo_auto = False
                        self.datos_modelo.clear()
                        self._reset_modelos()
                        self._reset_estado_juego()
                        self.stats_bala = EstadisticasBala()  # reset al iniciar sesión manual
                        esperando = False
                        break
                    if e.key == pygame.K_a:
                        if not self._modelo_activo_entrenado():
                            msg = f"Primero entrena con T. Activo: {self.tipo_modelo_activo}"
                        else:
                            self.modo_auto = True
                            self._reset_estado_juego()
                            esperando = False
                            break
                    if e.key == pygame.K_t:
                        ok1, info1 = self.entrenar_mlp()
                        ok2, info2 = self.entrenar_arbol()
                        if ok1 and ok2:
                            msg = f"{info1} | {info2}"
                        else:
                            msg = info1 if not ok1 else info2
                    if e.key == pygame.K_l:
                        if self.tipo_modelo_activo == TIPO_MLP:
                            self.tipo_modelo_activo = TIPO_ARBOL
                        else:
                            self.tipo_modelo_activo = TIPO_MLP
                        msg = f"Modelo activo: {self.tipo_modelo_activo}"
                    if e.key == pygame.K_c:
                        msg = self.exportar_datos_csv()
                    if e.key == pygame.K_i:
                        msg = self.cargar_csv()
                    if e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    if e.key == pygame.K_q:
                        self.corriendo = False
                        esperando = False
                        return

    # ═══════════════════════ render ═════════════════════════════════════════

    def _update_frame(self) -> None:
        # Fondo desplazable
        self.fondo_x1 -= self.fondo_speed
        self.fondo_x2 -= self.fondo_speed
        if self.fondo_x1 <= -self.w: self.fondo_x1 = self.w
        if self.fondo_x2 <= -self.w: self.fondo_x2 = self.w
        self.pantalla.blit(self.fondo_img, (self.fondo_x1, 0))
        self.pantalla.blit(self.fondo_img, (self.fondo_x2, 0))

        # Animación personaje
        self.frame_count += 1
        if self.frame_count >= self.frame_speed:
            self.current_frame = (self.current_frame + 1) % len(self.jugador_frames)
            self.frame_count   = 0

        frames = self.jugador_frames_agachado if self.agachado else self.jugador_frames
        self.pantalla.blit(frames[self.current_frame], (self.jugador.x, self.jugador.y))
        self.pantalla.blit(self.nave_img, (self.nave.x, self.nave.y))

        # Bala
        if self.bala_disparada:
            self.bala.x += self.velocidad_bala
        if self.bala.x < -self.bullet_size[0]:
            self.reset_bala()
        self.pantalla.blit(self.bala_img, (self.bala.x, self.bala.y))

        # Colisión
        if self.jugador.colliderect(self.bala):
            self._reset_estado_juego()

        # ── HUD Simplificado (Una sola línea de texto) ──────────────────────
        ps, pa, pe = self.stats_bala.porcentajes()
        total = self.stats_bala.total()
        
        if self.modo_auto and self._modelo_activo_entrenado():
            nombres_accion = {ACCION_QUIETO: "PARADO",
                              ACCION_SALTO:  "SALTO",
                              ACCION_AGACHADO: "AGACHADO"}
            accion_label = nombres_accion.get(self.ultima_accion_auto, "?")
            proba_str = (
                f" p_salto≈{self.ultima_proba_salto:.2f}"
                if self.ultima_proba_salto is not None else ""
            )
            hud_text = f"AUTO ({self.tipo_modelo_activo}): {accion_label}{proba_str} | Agachado->{pa:.0f}% | Parado->{pe:.0f}% | Salto ->{ps:.0f}% | n={total}"
        else:
            hud_text = f"Agachado->{pa:.0f}% | Parado->{pe:.0f}% | Salto ->{ps:.0f}% | n={total}"

        txt = self.fuente_chica.render(hud_text, True, self.AMARILLO)
        self.pantalla.blit(txt, (10, 10))

    # ═══════════════════════ loop principal ══════════════════════════════════
    def loop(self) -> None:
        reloj = pygame.time.Clock()
        self.mostrar_menu()

        while self.corriendo:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    self.corriendo = False
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        self.corriendo = False
                    elif e.key in (pygame.K_ESCAPE, pygame.K_p):
                        self._reset_estado_juego()
                        self.mostrar_menu()
                    elif e.key == pygame.K_f:
                        self._toggle_fullscreen()
                    elif e.key == pygame.K_l:
                        # Alternar modelo activo sin salir al menú
                        if self.tipo_modelo_activo == TIPO_MLP:
                            self.tipo_modelo_activo = TIPO_ARBOL
                        else:
                            self.tipo_modelo_activo = TIPO_MLP
                    elif e.key == pygame.K_SPACE and not self.modo_auto and self.en_suelo:
                        self.iniciar_salto()
                    elif e.key in (pygame.K_DOWN, pygame.K_s) and not self.modo_auto:
                        self.iniciar_agachado()
                elif e.type == pygame.KEYUP:
                    if e.key in (pygame.K_DOWN, pygame.K_s) and not self.modo_auto:
                        self.dejar_agachado()

            if not self.corriendo:
                break

            if self.modo_auto:
                accion = self.decision_auto()

                # ── Gestión del timer de agachado automático ──────────────
                if self.agachado and self.agachado_auto_timer > 0:
                    # Sigue agachado: descontar timer
                    self.agachado_auto_timer -= 1
                    if self.agachado_auto_timer == 0:
                        # Timer expiró → levantarse para poder repetir
                        self.dejar_agachado()
                elif accion == ACCION_SALTO and self.en_suelo and not self.agachado:
                    self.iniciar_salto()
                elif accion == ACCION_AGACHADO and self.en_suelo and not self.salto:
                    # Iniciar agachado con timer mínimo
                    self.iniciar_agachado()
                    self.agachado_auto_timer = AGACHADO_DURACION_AUTO
                # QUIETO: si el timer ya expiró y no hay nueva orden, simplemente
                # no se hace nada (el personaje ya está de pie tras el timer)
            else:
                self.registrar_decision_manual()

            if self.salto:
                self.manejar_salto()

            if not self.bala_disparada:
                self.disparar_bala()

            self._update_frame()
            pygame.display.flip()
            reloj.tick(45)

        pygame.quit()


def main() -> None:
    Juego().loop()


if __name__ == "__main__":
    main()
