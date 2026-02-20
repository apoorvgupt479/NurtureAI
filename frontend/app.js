/* ============================================================
   NurtureAI — Frontend Logic
   State management, form handling, API calls, result rendering
   ============================================================ */

const API = "";  // Same origin — Flask serves everything

const App = {
    // ---- State ----
    state: {
        parent: null,       // {name, General_Health, Sleep_Hours, ...}
        children: [],       // [{id, name, ageGroup, age, sex, lastResult}]
        selectedChild: null,// id of child being assessed
        chatHistory: [],    // chat history for Ask AI
        chatContextChildId: null // id of child currently selected in chat
    },

    // ================================================================
    // Initialization
    // ================================================================
    init() {
        const saved = localStorage.getItem("nurtureai_state");
        if (saved) {
            try {
                const parsed = JSON.parse(saved);
                this.state = { ...this.state, ...parsed };
                // Ensure arrays that might not exist in old states are initialized
                if (!this.state.chatHistory) this.state.chatHistory = [];
                if (!this.state.children) this.state.children = [];
            } catch (e) {
                localStorage.removeItem("nurtureai_state");
            }
        }

        if (this.state.parent) {
            this.showScreen("screen-dashboard");
            this.renderDashboard();
        } else {
            this.showScreen("screen-welcome");
        }

        // Kick off model loading in background
        this.loadModels();

        // Auto-calculate BMI when height/weight change
        const hEl = document.getElementById("ch-height");
        const wEl = document.getElementById("ch-weight");
        if (hEl && wEl) {
            hEl.addEventListener("input", () => this.autoBMI());
            wEl.addEventListener("input", () => this.autoBMI());
        }

        // Parent BMI
        const pHEl = document.getElementById("parent-bmi"); // Using a simplified version for the parent just entering BMI directly for now, or could compute if we added height/weight. We added direct input.
        
        this.checkApiKey();
    },

    async checkApiKey() {
        try {
            const res = await fetch(API + "/api/get-api-key");
            const data = await res.json();
            const statusEl = document.getElementById("key-status");
            if (data.has_key && statusEl) {
                statusEl.textContent = "Key loaded: " + data.masked_key;
                statusEl.style.color = "var(--success)";
            }
        } catch(e) {}
    },

    save() {
        localStorage.setItem("nurtureai_state", JSON.stringify(this.state));
    },

    // ================================================================
    // Screen Navigation
    // ================================================================
    showScreen(id) {
        document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
        const screen = document.getElementById(id);
        if (screen) screen.classList.add("active");
        window.scrollTo(0, 0);

        // Handle Chat AI button visibility
        const chatFab = document.getElementById("chat-fab");
        if (chatFab) {
            if (this.state.parent && this.state.children.length > 0) {
                chatFab.style.display = "flex";
            } else {
                chatFab.style.display = "none";
            }
        }
    },

    // ================================================================
    // Models (background loading)
    // ================================================================
    async loadModels() {
        try {
            await fetch(API + "/api/load-models", { method: "POST" });
        } catch (e) {
            // Server might not be running — that's OK for now
            console.log("Model loading request failed (server may be offline):", e.message);
        }
    },

    // ================================================================
    // Welcome → Start Setup
    // ================================================================
    startSetup() {
        this.showScreen("screen-parent");
    },

    // ================================================================
    // Parent Wizard
    // ================================================================
    parentStep(step) {
        // Validate current step
        if (step === 2) {
            const name = document.getElementById("parent-name").value.trim();
            if (!name) { this.toast("Please enter your name"); return; }
        }

        // Switch step
        document.querySelectorAll("#screen-parent .wizard-step").forEach(s => s.classList.remove("active"));
        const target = document.getElementById("parent-step-" + step);
        if (target) target.classList.add("active");

        // Update step indicators
        document.querySelectorAll("#parent-steps .step").forEach(s => {
            const n = parseInt(s.dataset.step);
            s.classList.toggle("active", n === step);
            s.classList.toggle("done", n < step);
        });
    },

    saveParent() {
        const name = document.getElementById("parent-name").value.trim();
        if (!name) { this.toast("Please enter your name"); this.parentStep(1); return; }

        const healthEl = document.querySelector('input[name="health"]:checked');

        this.state.parent = {
            name: name,
            General_Health: parseInt(healthEl ? healthEl.value : 3),
            Sleep_Hours: parseInt(document.getElementById("parent-sleep").value) || 7,
            Exercise_Any: parseInt(document.getElementById("parent-exercise").value) || 1,
            Income_Level: parseInt(document.getElementById("parent-income").value) || 6,
            Marital_Status: parseInt(document.getElementById("parent-marital").value) || 1,
            Smoked_100_Cigs: parseInt(document.getElementById("parent-smoke").value) || 2,
            Physical_Health_Days: parseInt(document.getElementById("parent-phys-days").value) || 0,
            Mental_Health_Days: parseInt(document.getElementById("parent-mental-days").value) || 0,
            Depression_Diagnosis: parseInt(document.getElementById("parent-depression").value) || 2,
            BMI_Indicator: parseFloat(document.getElementById("parent-bmi").value) || 24.5,
            Alcohol_Days_Monthly: parseInt(document.getElementById("parent-alcohol").value) || 0
        };

        this.save();
        this.showScreen("screen-dashboard");
        this.renderDashboard();
        this.toast("Profile saved! 🎉");

        // Also run parent assessment in background
        this.runParentAssessment();
    },

    async runParentAssessment() {
        try {
            const parentData = { ...this.state.parent };
            delete parentData.name;
            const res = await fetch(API + "/api/parent-assessment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(parentData)
            });
            const result = await res.json();
            this.state.parent.assessment = result;
            this.save();
            this.renderDashboard();
        } catch (e) {
            console.log("Parent assessment failed:", e.message);
        }
    },

    // ================================================================
    // Dashboard
    // ================================================================
    renderDashboard() {
        if (!this.state.parent) return;

        const greeting = document.getElementById("dash-greeting");
        greeting.textContent = `Welcome, ${this.state.parent.name}! 👋`;

        // Parent card
        const badge = document.getElementById("parent-badge");
        const status = document.getElementById("parent-card-status");
        const pName = document.getElementById("parent-card-name");
        pName.textContent = this.state.parent.name;

        if (this.state.parent.assessment) {
            const pred = this.state.parent.assessment.prediction || "";
            status.textContent = pred;
            if (pred.toLowerCase().includes("healthy")) {
                badge.textContent = "✓";
                badge.className = "badge badge-green";
            } else if (pred.toLowerCase().includes("moderate") || pred.toLowerCase().includes("mild")) {
                badge.textContent = "!";
                badge.className = "badge badge-yellow";
            } else if (pred.toLowerCase().includes("high") || pred.toLowerCase().includes("severe")) {
                badge.textContent = "⚠";
                badge.className = "badge badge-red";
            } else {
                badge.textContent = "✓";
                badge.className = "badge badge-green";
                status.textContent = "Profile complete";
            }
        } else {
            status.textContent = "Profile complete";
        }

        // Children list
        const list = document.getElementById("children-list");
        const empty = document.getElementById("no-children");
        list.innerHTML = "";

        if (this.state.children.length === 0) {
            empty.style.display = "block";
        } else {
            empty.style.display = "none";
            this.state.children.forEach(child => {
                const icon = child.ageGroup === "infant" ? "👶" : "🧒";
                const ageText = child.ageGroup === "infant" ? "Under 1 year" : `${child.age} years old`;
                const lastResult = child.lastResult ? this.getResultBadge(child) : '<span class="child-action">Assess →</span>';

                const div = document.createElement("div");
                div.className = "child-card";
                div.onclick = (e) => {
                    if (e.target.closest('.celiac-btn')) return;
                    this.startChildAssessment(child.id);
                };
                div.innerHTML = `
                    <span class="child-icon">${icon}</span>
                    <div class="child-info">
                        <div class="child-name">${child.name}</div>
                        <div class="child-age">${ageText} · ${child.sex === 0 ? "Female" : "Male"}</div>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:4px; align-items:flex-end;">
                        ${lastResult}
                        <button class="btn btn-ghost celiac-btn" style="padding:4px 8px; font-size:0.8rem" onclick="App.startCeliacAssessment('${child.id}')">Celiac Screen</button>
                    </div>
                `;
                list.appendChild(div);
            });
        }
    },

    getResultBadge(child) {
        if (!child.lastResult) return "";
        const r = child.lastResult;

        // For nurture model
        if (r.prediction && r.prediction.risk_label) {
            const label = r.prediction.risk_label;
            if (label.includes("Healthy") || label.includes("None")) return '<span class="badge badge-green">✓</span>';
            if (label.includes("Mild") || label.includes("Moderate")) return '<span class="badge badge-yellow">!</span>';
            return '<span class="badge badge-red">⚠</span>';
        }

        // For child_mortality: 1 = Low Risk (Alive), 0 = High Risk (Dead)
        if (r.prediction !== undefined && r.prediction_status === undefined) {
            return r.prediction === 1
                ? '<span class="badge badge-green">✓</span>'
                : '<span class="badge badge-red">⚠</span>';
        }

        return '<span class="child-action">Assess →</span>';
    },

    // ================================================================
    // Add Child
