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
