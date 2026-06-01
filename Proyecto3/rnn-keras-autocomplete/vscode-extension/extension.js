const vscode = require("vscode");
const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");

let proc, rl, pending = new Map(), reqId = 0;

function serverScript() {
    const cfg = vscode.workspace.getConfiguration("rnnKeras");
    const custom = cfg.get("serverScript");
    if (custom) return custom;
    // __dirname es la carpeta de la extensión: rnn-keras-autocomplete/vscode-extension
    // Por lo tanto, subir un nivel (..) nos lleva a rnn-keras-autocomplete/
    return path.join(__dirname, "..", "server_stdio.py");
}

function request(method, fields) {
    return new Promise((resolve, reject) => {
        if (!proc) {
            const py = vscode.workspace.getConfiguration("rnnKeras").get("pythonPath") || "python";
            const script = serverScript();
            
            console.log(`[RNN Extension]: Iniciando servidor con ${py} en la ruta ${script}`);
            
            proc = spawn(py, [script], { 
                cwd: path.dirname(script), 
                stdio: ["pipe", "pipe", "pipe"] 
            });

            rl = readline.createInterface({ input: proc.stdout });
            rl.on("line", (line) => {
                try {
                    const msg = JSON.parse(line);
                    if (pending.has(msg._id)) {
                        pending.get(msg._id)(msg);
                        pending.delete(msg._id);
                    }
                } catch (e) {
                    console.error("[RNN Extension Error]: Fallo al parsear línea del servidor:", line);
                }
            });

            proc.on("error", (err) => {
                vscode.window.showErrorMessage("Error al iniciar el servidor de autocompletado (¿Python mal configurado?): " + err.message);
                proc = null;
                reject(err);
            });

            proc.stderr.on("data", (data) => {
                console.log(`[RNN Server Log]: ${data.toString().trim()}`);
            });

            proc.on("close", (code) => {
                console.log(`[RNN Extension]: Servidor cerrado con código ${code}`);
                proc = null;
            });
        }
        
        const id = ++reqId;
        const timer = setTimeout(() => {
            pending.delete(id);
            reject(new Error("tiempo de espera agotado (timeout)"));
        }, 20000);
        
        pending.set(id, (msg) => {
            clearTimeout(timer);
            msg.ok ? resolve(msg) : reject(new Error(msg.error || "error"));
        });
        
        proc.stdin.write(JSON.stringify({ method, _id: id, ...fields }) + "\n");
    });
}

async function completeLine() {
    const ed = vscode.window.activeTextEditor;
    if (!ed) return;
    const pos = ed.selection.active;
    const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
    const maxNew = vscode.workspace.getConfiguration("rnnKeras").get("maxNew") || 60;
    
    vscode.window.setStatusBarMessage("RNN: Completando...", 2000);
    
    try {
        const res = await request("complete", { prefix, max_new: maxNew, temperature: 0.2 });
        const suffix = res.text.slice(prefix.length);
        await ed.edit((eb) => eb.insert(pos, suffix));
    } catch (err) {
        vscode.window.showWarningMessage("Autocompletado fallido: " + err.message);
    }
}

async function showSuggestions() {
    const ed = vscode.window.activeTextEditor;
    if (!ed) return;
    const pos = ed.selection.active;
    const prefix = ed.document.lineAt(pos.line).text.slice(0, pos.character);
    
    vscode.window.setStatusBarMessage("RNN: Buscando sugerencias...", 2000);
    
    try {
        const res = await request("suggest", { prefix, n: 5 });
        if (!res.items || res.items.length === 0) {
            vscode.window.showInformationMessage("No hay sugerencias disponibles");
            return;
        }
        const pick = await vscode.window.showQuickPick(res.items, { placeHolder: "Sugerencias de código RNN" });
        if (!pick) return;
        await ed.edit((eb) => eb.insert(pos, pick.slice(prefix.length)));
    } catch (err) {
        vscode.window.showWarningMessage("Sugerencias fallidas: " + err.message);
    }
}

function activate(ctx) {
    ctx.subscriptions.push(
        vscode.commands.registerCommand("rnnKeras.complete", completeLine),
        vscode.commands.registerCommand("rnnKeras.suggest", showSuggestions)
    );
    console.log("[RNN Extension]: Extensión activada con éxito.");
}

function deactivate() {
    if (proc) {
        proc.kill();
        console.log("[RNN Extension]: Proceso del servidor eliminado.");
    }
}

module.exports = { activate, deactivate };
