import { app } from "../../scripts/app.js";

class WanPromptBatchUI {
    constructor(node, textWidget) {
        this.node = node;
        this.textWidget = textWidget;
        // Parse current value or init default
        let data = [];
        try {
            data = JSON.parse(this.textWidget.value);
            if (!Array.isArray(data)) data = [];
        } catch(e) {
            data = [];
        }
        if (data.length === 0) {
            data.push({ text: "", curve: "linear" });
        }
        this.items = data;
        
        this.root = document.createElement("div");
        this.root.style.cssText = `
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            gap: 8px;
            background: #141414;
            border-radius: 6px;
            padding: 8px;
            box-sizing: border-box;
            overflow-y: auto;
        `;
        
        this.listContainer = document.createElement("div");
        this.listContainer.style.cssText = `
            display: flex;
            flex-direction: column;
            gap: 6px;
        `;
        
        this.addBtn = document.createElement("button");
        this.addBtn.textContent = "➕ Add Prompt";
        this.addBtn.style.cssText = `
            padding: 6px;
            background: #244a24;
            border: 1px solid #4a6;
            color: #eee;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 4px;
        `;
        this.addBtn.onclick = () => {
            this.items.push({ text: "", curve: "linear" });
            this.render();
            this.sync();
        };
        
        this.root.appendChild(this.listContainer);
        this.root.appendChild(this.addBtn);
        this.render();
    }
    
    render() {
        this.listContainer.innerHTML = "";
        this.items.forEach((item, index) => {
            const itemDiv = document.createElement("div");
            itemDiv.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 4px;
                padding: 6px;
                background: #1f1f1f;
                border: 1px solid #333;
                border-radius: 4px;
            `;
            
            const header = document.createElement("div");
            header.style.cssText = "display: flex; justify-content: space-between; align-items: center;";
            
            const title = document.createElement("span");
            title.textContent = `Prompt #${index + 1}`;
            title.style.color = "#aaa";
            title.style.fontSize = "12px";
            
            const rightControls = document.createElement("div");
            rightControls.style.display = "flex";
            rightControls.style.gap = "4px";
            
            const curveSelect = document.createElement("select");
            curveSelect.style.cssText = "background: #222; color: #eee; border: 1px solid #444; border-radius: 3px; font-size: 11px; padding: 2px;";
            const curves = ["linear", "ease-in", "ease-out", "ease-in-out"];
            curves.forEach(c => {
                const opt = document.createElement("option");
                opt.value = c;
                opt.textContent = c;
                if (item.curve === c) opt.selected = true;
                curveSelect.appendChild(opt);
            });
            curveSelect.onchange = (e) => {
                this.items[index].curve = e.target.value;
                this.sync();
            };
            
            const delBtn = document.createElement("button");
            delBtn.textContent = "×";
            delBtn.style.cssText = "background: #4a2222; border: 1px solid #a44; color: #eee; border-radius: 3px; cursor: pointer; padding: 0 4px;";
            delBtn.onclick = () => {
                this.items.splice(index, 1);
                this.render();
                this.sync();
            };
            
            rightControls.appendChild(curveSelect);
            if (this.items.length > 1) {
                rightControls.appendChild(delBtn);
            }
            
            header.appendChild(title);
            header.appendChild(rightControls);
            
            const textArea = document.createElement("textarea");
            textArea.value = item.text;
            textArea.rows = 3;
            textArea.style.cssText = "width: 100%; box-sizing: border-box; background: #222; border: 1px solid #444; color: #eee; border-radius: 3px; padding: 4px; font-family: monospace; resize: vertical;";
            textArea.oninput = (e) => {
                this.items[index].text = e.target.value;
                this.sync();
            };
            
            itemDiv.appendChild(header);
            itemDiv.appendChild(textArea);
            this.listContainer.appendChild(itemDiv);
        });
    }
    
    sync() {
        this.textWidget.value = JSON.stringify(this.items);
    }
}

app.registerExtension({
    name: "Comfy.WanPromptBatch",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "WanPromptBatch") {
            const origOnCreate = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origOnCreate?.apply(this, arguments);
                let widget = this.widgets?.find((w) => w.name === "prompts_data");
                if (!widget) {
                    widget = this.addWidget("text", "prompts_data", "[]", () => {}, { multiline: false });
                }
                widget.hidden = true;
                widget.serialize = true;
                
                this.__promptUI = new WanPromptBatchUI(this, widget);
                
                // standard comfy nodes usually pass dom widgets this way
                this.addDOMWidget("wan_prompt_batch_ui", "custom", this.__promptUI.root);
                
                const onResize = this.onResize;
                this.onResize = function() {
                    if (onResize) onResize.apply(this, arguments);
                    // Ensure minimum size
                    if (this.size[0] < 350) this.size[0] = 350;
                };
                
                this.setSize([400, 300]);
                return r;
            };
        }
    }
});
