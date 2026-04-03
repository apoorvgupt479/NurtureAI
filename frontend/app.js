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
    // ================================================================
    addChild() {
        const name = document.getElementById("child-name").value.trim();
        if (!name) { this.toast("Please enter child's name"); return; }

        const ageGroup = document.getElementById("child-age-group").value;
        const age = ageGroup === "older" ? parseInt(document.getElementById("child-age").value) || 5 : 0;
        const sex = parseInt(document.getElementById("child-sex").value) || 0;

        const child = {
            id: Date.now().toString(),
            name, ageGroup, age, sex,
            lastResult: null
        };

        this.state.children.push(child);
        this.save();

        // Start assessment for this child
        this.startChildAssessment(child.id);
    },

    startChildAssessment(childId) {
        this.state.selectedChild = childId;
        const child = this.state.children.find(c => c.id === childId);
        if (!child) return;

        if (child.ageGroup === "infant") {
            this.showScreen("screen-infant-form");
            const title = document.getElementById("infant-title-1");
            if (title) title.textContent = `${child.name} — Birth & Family`;
            this.infantStep(1);
        } else {
            this.showScreen("screen-child-form");
            const title = document.getElementById("child-title-1");
            if (title) title.textContent = `${child.name} — Physical Measurements`;
            this.childStep(1);
        }
    },

    // ================================================================
    // Infant Assessment Form (< 1 year)
    // ================================================================
    infantStep(step) {
        document.querySelectorAll("#screen-infant-form .wizard-step").forEach(s => s.classList.remove("active"));
        const target = document.getElementById("infant-step-" + step);
        if (target) target.classList.add("active");

        document.querySelectorAll("#infant-steps .step").forEach(s => {
            const n = parseInt(s.dataset.step);
            s.classList.toggle("active", n === step);
            s.classList.toggle("done", n < step);
        });
    },

    async submitInfant() {
        this.showLoading("Analyzing infant health data...");

        const data = {
            Toilet_Facility:       parseInt(document.getElementById("inf-toilet").value),
            Child_under5:          parseInt(document.getElementById("inf-under5").value) || 0,
            Tot_child_born:        parseInt(document.getElementById("inf-children").value) || 1,
            Sons_died:             parseInt(document.getElementById("inf-sons-died").value) || 0,
            Daughters_died:        parseInt(document.getElementById("inf-daughters-died").value) || 0,
            Curr_Preg:             parseInt(document.getElementById("inf-pregnant").value) || 0,
            Curr_BrstFeed:         parseInt(document.getElementById("inf-breastfeed").value),
            ChildFood_bottle:      parseInt(document.getElementById("inf-bottle").value),
            Resp_height:           parseFloat(document.getElementById("inf-height").value) || 155,
            HealthInsurance:       parseInt(document.getElementById("inf-insurance").value),
            B_ChildTwin:           parseInt(document.getElementById("inf-twin").value),
            First3Day_fruitJuice:  document.getElementById("inf-fruitjuice").checked ? 1 : 0,
            HepatitisB_atBirth:    document.getElementById("inf-hepb").checked ? 1 : 0,
            ShortBreaths:          parseInt(document.getElementById("inf-breath").value),
            VitaminA:              document.getElementById("inf-vita").checked ? 1 : 0,
            IronPill:              document.getElementById("inf-iron").checked ? 1 : 0,
            IntestinalDrug:        document.getElementById("inf-deworm").checked ? 1 : 0,
            ultrasound:            parseInt(document.getElementById("inf-ultrasound").value),
            MMR:                   document.getElementById("inf-mmr").checked ? 1 : 0,
            delivery_place:        parseInt(document.getElementById("inf-private").value) ? "private" : "government",
            Water_Source_Other:    parseInt(document.getElementById("inf-water").value) || 0,
            DPT_full:              document.getElementById("inf-dpt").checked ? 1 : 0,
            MEASLES_full:          document.getElementById("inf-measles").checked ? 1 : 0,
        };

        // State
        const stateVal = document.getElementById("inf-state").value;
        if (stateVal) data.state = stateVal;

        try {
            const res = await fetch(API + "/api/child-infant", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const result = await res.json();

            // Save result to child
            const child = this.state.children.find(c => c.id === this.state.selectedChild);
            if (child) { child.lastResult = result; this.save(); }

            this.hideLoading();
            this.renderInfantResults(result, child);
        } catch (e) {
            this.hideLoading();
            this.toast("Error: " + e.message);
        }
    },

    // ================================================================
    // Child Assessment Form (>= 1 year)
    // ================================================================
    childStep(step) {
        document.querySelectorAll("#screen-child-form .wizard-step").forEach(s => s.classList.remove("active"));
        const target = document.getElementById("child-step-" + step);
        if (target) target.classList.add("active");

        document.querySelectorAll("#child-steps .step").forEach(s => {
            const n = parseInt(s.dataset.step);
            s.classList.toggle("active", n === step);
            s.classList.toggle("done", n < step);
        });
    },

    autoBMI() {
        const h = parseFloat(document.getElementById("ch-height").value);
        const w = parseFloat(document.getElementById("ch-weight").value);
        if (h && w && h > 0) {
            const bmi = (w / ((h / 100) ** 2)).toFixed(1);
            document.getElementById("ch-bmi").value = bmi;
        }
    },

    async submitChild() {
        this.showLoading("Running health assessment...");

        const child = this.state.children.find(c => c.id === this.state.selectedChild);
        if (!child) { this.hideLoading(); return; }

        const data = {
            "Basic_Demos-Age": child.age,
            "Basic_Demos-Sex": child.sex,
        };

        // Physical (optional — model fills medians for missing)
        const addIfPresent = (elId, key) => {
            const v = parseFloat(document.getElementById(elId).value);
            if (!isNaN(v) && v > 0) data[key] = v;
        };

        addIfPresent("ch-height", "Physical-Height");
        addIfPresent("ch-weight", "Physical-Weight");
        addIfPresent("ch-bmi", "Physical-BMI");
        addIfPresent("ch-waist", "Physical-Waist_Circumference");
        addIfPresent("ch-hr", "Physical-HeartRate");
        addIfPresent("ch-sbp", "Physical-Systolic_BP");
        addIfPresent("ch-dbp", "Physical-Diastolic_BP");
        addIfPresent("ch-sds", "SDS-SDS_Total_T");
        addIfPresent("ch-fit-stage", "Fitness_Endurance-Max_Stage");
        addIfPresent("ch-fit-time", "Fitness_Endurance-Time_Mins");
        addIfPresent("ch-fat", "BIA-BIA_Fat");
        addIfPresent("ch-ffm", "BIA-BIA_FFM");
        addIfPresent("ch-smm", "BIA-BIA_SMM");

        // PAQ: use PAQ_C for children <=12, PAQ_A for teens
        const paq = parseFloat(document.getElementById("ch-paq").value);
        if (!isNaN(paq) && paq > 0) {
            if (child.age <= 12) data["PAQ_C-PAQ_C_Total"] = paq;
            else data["PAQ_A-PAQ_A_Total"] = paq;
        }

        try {
            const res = await fetch(API + "/api/child-health", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const result = await res.json();

            if (child) { child.lastResult = result; this.save(); }
            this.hideLoading();
            this.renderChildResults(result, child);
        } catch (e) {
            this.hideLoading();
            this.toast("Error: " + e.message);
        }
    },

    // ================================================================
    // Results Rendering
    // ================================================================
    renderInfantResults(result, child) {
        const container = document.getElementById("results-container");
        const prediction = result.prediction;
        const prob = result.probability_class_1;

        // Class 1 = Survival/Healthy, Class 0 = Mortality/Risk
        const isHealthy = prediction === 1;
        const riskProb = 1 - prob; // Risk is the inverse of survival probability

        const icon = isHealthy ? "✅" : "⚠️";
        const title = isHealthy ? "Low Risk" : "Elevated Risk Detected";
        const alertClass = isHealthy ? "alert-success" : "alert-danger";
        const alertMsg = isHealthy
            ? "Based on the information provided, your infant appears to be at <strong>low risk</strong>. Keep up the great care!"
            : "Some risk factors were detected. Please consult with your pediatrician for a thorough evaluation.";

        container.innerHTML = `
            <div class="result-header">
                <div class="result-icon">${icon}</div>
                <h2>${child ? child.name + " — " : ""}${title}</h2>
                <p class="result-subtitle">Infant Health Assessment</p>
            </div>

            <div class="result-alert ${alertClass}">${alertMsg}</div>

            ${prob !== undefined ? `
            <div class="result-card">
                <h3>📊 Risk Probability</h3>
                <div class="result-row">
                    <span class="result-label">Risk Score</span>
                    <span class="result-value">${(riskProb * 100).toFixed(1)}%</span>
                </div>
                <div class="result-bar">
                    <div class="result-bar-fill ${riskProb > 0.5 ? 'bar-red' : riskProb > 0.2 ? 'bar-yellow' : 'bar-green'}"
                         style="width:${(riskProb * 100).toFixed(1)}%"></div>
                </div>
            </div>` : ""}

            <button class="btn btn-primary btn-block" onclick="App.showScreen('screen-dashboard'); App.renderDashboard();">
                ← Back to Dashboard
            </button>
        `;

        this.showScreen("screen-results");
    },

    renderChildResults(result, child) {
        const container = document.getElementById("results-container");

        if (result.status === "error" || result.code === 500) {
            container.innerHTML = `
                <div class="result-header">
                    <div class="result-icon">❌</div>
                    <h2>Assessment Failed</h2>
                    <p class="result-subtitle">${result.message || "Unknown error"}</p>
                </div>
                <button class="btn btn-primary btn-block" onclick="App.showScreen('screen-dashboard')">← Back</button>
            `;
            this.showScreen("screen-results");
            return;
        }

        const pred = result.prediction || {};
        const sii = pred.sii !== undefined ? pred.sii : -1;
        const riskLabel = pred.risk_label || "Unknown";
        const behaviorScore = pred.behavior_score || 0;
        const probabilities = pred.probabilities || {};

        const icons = { 0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴" };
        const alertClasses = { 0: "alert-success", 1: "alert-warning", 2: "alert-warning", 3: "alert-danger" };
        const icon = icons[sii] || "❓";

        const messages = {
            0: "Your child shows a <strong>healthy</strong> profile. Keep nurturing their wellbeing!",
            1: "Some <strong>mild</strong> risk factors detected. Small lifestyle adjustments can make a big difference.",
            2: "There are <strong>moderate</strong> concerns. Consider consulting a pediatrician for personalized guidance.",
            3: "The assessment shows <strong>significant</strong> risk factors. We strongly recommend professional consultation."
        };

        let probHTML = "";
        for (const [label, prob] of Object.entries(probabilities)) {
            const pct = (prob * 100).toFixed(1);
            const barColor = label.includes("Healthy") ? "bar-green" : label.includes("Mild") ? "bar-yellow" :
                             label.includes("Moderate") ? "bar-yellow" : "bar-red";
            probHTML += `
                <div class="result-row">
                    <span class="result-label">${label.replace(/sii=\d+ \(/, "").replace(")", "")}</span>
                    <span class="result-value">${pct}%</span>
                </div>
                <div class="result-bar"><div class="result-bar-fill ${barColor}" style="width:${pct}%"></div></div>
            `;
        }

        container.innerHTML = `
            <div class="result-header">
                <div class="result-icon">${icon}</div>
                <h2>${child ? child.name + " — " : ""}${riskLabel}</h2>
                <p class="result-subtitle">Child Health & Behaviour Assessment</p>
            </div>

            <div class="result-alert ${alertClasses[sii] || 'alert-warning'}">
                ${messages[sii] || "Assessment complete."}
            </div>

            ${pred.behavior_score !== undefined ? `
            <div class="result-card">
                <h3>🧠 Behavior Score</h3>
                <div class="result-row">
                    <span class="result-label">Lifestyle Score</span>
                    <span class="result-value">${behaviorScore} / 100</span>
                </div>
                <div class="result-bar">
                    <div class="result-bar-fill ${behaviorScore >= 60 ? 'bar-green' : behaviorScore >= 35 ? 'bar-yellow' : 'bar-red'}"
                         style="width:${behaviorScore}%"></div>
                </div>
            </div>
            ` : ""}

            ${probHTML ? `<div class="result-card"><h3>📊 Risk Breakdown</h3>${probHTML}</div>` : ""}

            <button class="btn btn-primary btn-block" onclick="App.showScreen('screen-dashboard'); App.renderDashboard();">
                ← Back to Dashboard
            </button>
        `;

        this.showScreen("screen-results");
    },

    // ================================================================
    // Utility Helpers
    // ================================================================
    toggleSelect(btn) {
        const field = btn.dataset.field;
        const value = btn.dataset.value;
        const group = btn.parentElement;
        group.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(field).value = value;

        // Show/hide exact age field
        if (field === "child-age-group") {
            const ageGroup = document.getElementById("child-exact-age-group");
            if (ageGroup) ageGroup.style.display = value === "older" ? "block" : "none";
        }
    },

    adjustStepper(id, delta) {
        const el = document.getElementById(id);
        const min = parseInt(el.min) || 0;
        const max = parseInt(el.max) || 100;
        let val = parseInt(el.value) || 0;
        val = Math.max(min, Math.min(max, val + delta));
        el.value = val;
    },

    showLoading(text) {
        document.getElementById("loading-text").textContent = text || "Processing...";
        document.getElementById("loading-overlay").classList.add("active");
    },

    hideLoading() {
        document.getElementById("loading-overlay").classList.remove("active");
    },

    toast(msg) {
        const el = document.getElementById("toast");
        el.textContent = msg;
        el.classList.add("show");
        setTimeout(() => el.classList.remove("show"), 2500);
    },

    resetApp() {
        if (confirm("Reset all data? This will clear your profile and children.")) {
            localStorage.removeItem("nurtureai_state");
            this.state = { parent: null, children: [], selectedChild: null, chatHistory: [] };
            this.showScreen("screen-welcome");
            this.toast("App reset complete");
        }
    },

    // ================================================================
    // Celiac Disease Form
    // ================================================================
    startCeliacAssessment(childId) {
        this.state.selectedChild = childId;
        const child = this.state.children.find(c => c.id === childId);
        if (!child) return;
        this.showScreen("screen-celiac-form");
        this.celiacStep(1);
    },

    celiacStep(step) {
        document.querySelectorAll("#screen-celiac-form .wizard-step").forEach(s => s.classList.remove("active"));
        const target = document.getElementById("celiac-step-" + step);
        if (target) target.classList.add("active");

        document.querySelectorAll("#celiac-steps .step").forEach(s => {
            const n = parseInt(s.dataset.step);
            s.classList.toggle("active", n === step);
            s.classList.toggle("done", n < step);
        });
    },

    showDiabetesType(show) {
        document.getElementById("cel-diabetes-type-group").style.display = show ? "block" : "none";
    },

    async submitCeliac() {
        this.showLoading("Running celiac screening...");
        const child = this.state.children.find(c => c.id === this.state.selectedChild);
        if (!child) { this.hideLoading(); return; }

        const data = {
            "Age": child.age,
            "Gender": child.sex === 0 ? "Female" : "Male",
            "Diabetes": document.getElementById("cel-diabetes").value,
            "Diabetes Type": document.getElementById("cel-diabetes").value === "Yes" ? document.getElementById("cel-diabetes-type").value : "Unknown",
            "Diarrhoea": document.getElementById("cel-diarrhoea").checked ? "Yes" : "No",
            "Abdominal": document.getElementById("cel-abdominal").checked ? "Yes" : "No",
            "Short_Stature": document.getElementById("cel-short-stature").checked ? "Yes" : "No",
            "Sticky_Stool": document.getElementById("cel-sticky-stool").checked ? "Yes" : "No",
            "Weight_loss": document.getElementById("cel-weight-loss").checked ? "Yes" : "No",
            "IgA": parseFloat(document.getElementById("cel-iga").value) || 0.0,
            "IgG": parseFloat(document.getElementById("cel-igg").value) || 0.0,
            "IgM": parseFloat(document.getElementById("cel-igm").value) || 0.0
        };

        try {
            const res = await fetch(API + "/api/celiac-check", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data)
            });
            const result = await res.json();
            this.hideLoading();
            this.renderCeliacResults(result, child);
        } catch (e) {
            this.hideLoading();
            this.toast("Error: " + e.message);
        }
    },

    renderCeliacResults(result, child) {
        const container = document.getElementById("results-container");

        if (result.status === "error" || result.code === 500) {
            container.innerHTML = `
                <div class="result-header">
                    <div class="result-icon">❌</div>
                    <h2>Assessment Failed</h2>
                    <p class="result-subtitle">${result.message || "Unknown error"}</p>
                </div>
                <button class="btn btn-primary btn-block" onclick="App.showScreen('screen-dashboard')">← Back</button>
            `;
            this.showScreen("screen-results");
            return;
        }

        const isPositive = result.prediction === 1;
        const icon = isPositive ? "⚠️" : "✅";
        const title = isPositive ? "High Risk Detected" : "Low Risk";
        const alertClass = isPositive ? "alert-danger" : "alert-success";
        const alertMsg = isPositive 
            ? "The screening indicates a <strong>high risk</strong> of Celiac Disease based on symptoms and available lab markers. A formal medical diagnosis (such as an endoscopy) is strongly recommended."
            : "The screening indicates a <strong>low risk</strong> of Celiac Disease. However, if symptoms persist, consult a doctor.";

        container.innerHTML = `
            <div class="result-header">
                <div class="result-icon">${icon}</div>
                <h2>${child ? child.name + " — " : ""}${title}</h2>
                <p class="result-subtitle">Celiac Disease Screening</p>
            </div>
            <div class="result-alert ${alertClass}">${alertMsg}</div>
            <button class="btn btn-primary btn-block" onclick="App.showScreen('screen-dashboard'); App.renderDashboard();">
                ← Back to Dashboard
            </button>
        `;
        this.showScreen("screen-results");
    },

    // ================================================================
    // Settings (API Key)
    // ================================================================
    openSettings() {
        document.getElementById("settings-modal").classList.add("active");
    },
    closeSettings() {
        document.getElementById("settings-modal").classList.remove("active");
    },
    async saveApiKey() {
        const key = document.getElementById("settings-api-key").value.trim();
        if (!key) return;
        try {
            await fetch(API + "/api/save-api-key", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({api_key: key})
            });
            this.toast("API key saved");
            document.getElementById("settings-api-key").value = "";
            this.checkApiKey();
            this.closeSettings();
        } catch(e) {
            this.toast("Error saving API key");
        }
    },

    // ================================================================
    // Chatbot
    // ================================================================
    toggleChat() {
        const drawer = document.getElementById("chat-drawer");
        if (drawer.classList.contains("active")) {
            drawer.classList.remove("active");
        } else {
            drawer.classList.add("active");
            
            // If chat is empty, initialize it based on children
            if (this.state.chatHistory.length === 0) {
                if (this.state.children.length === 0) {
                    this.state.chatHistory.push({role: "ai", text: "Please add a child profile first from the dashboard so I can help you with specific medical questions."});
                } else if (this.state.children.length === 1) {
                    this.state.chatContextChildId = this.state.children[0].id;
                    this.state.chatHistory.push({role: "ai", text: `Hi! I'm your NurtureAI assistant. I'm ready to answer questions about ${this.state.children[0].name}. How can I help?`});
                } else {
                    // Multiple children, ask to select
                    this.state.chatContextChildId = null;
                }
            }
            
            this.renderMessages();
            setTimeout(()=> document.getElementById("chat-input").focus(), 100);
        }
    },

    setChatContext(childId) {
        const child = this.state.children.find(c => c.id === childId);
        if (child) {
            this.state.chatContextChildId = childId;
            this.state.chatHistory.push({role: "ai", text: `Got it. Let's talk about ${child.name}. What questions do you have?`});
            this.save();
            this.renderMessages();
        }
    },

    renderMessages() {
        const container = document.getElementById("chat-messages");
        container.innerHTML = "";
        
        // Show selection buttons if multiple children and none selected
        if (this.state.children.length > 1 && !this.state.chatContextChildId) {
            container.innerHTML += `
                <div class="chat-msg ai">
                    <div class="chat-bubble">Hi! I'm your NurtureAI assistant. Which child would you like to talk about today?</div>
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:8px;">
                    ${this.state.children.map(c => `<button class="btn btn-ghost" style="padding:6px 12px; font-size:0.85rem;" onclick="App.setChatContext('${c.id}')">${c.name}</button>`).join('')}
                </div>
            `;
        }

        this.state.chatHistory.forEach(msg => {
            container.innerHTML += `
                <div class="chat-msg ${msg.role === 'user' ? 'user' : 'ai'}">
                    <div class="chat-bubble">${msg.text}</div>
                </div>
            `;
        });
        
        // Disable input if no child context
        const input = document.getElementById("chat-input");
