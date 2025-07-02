import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Analytical SOP Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    
    header[data-testid="stHeader"] {
        display: none;
    }
    
    .stDeployButton {
        display: none;
    }
    
    h1 { 
        font-size: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    h2, h3 {
        font-size: 1rem;
        margin: 0.3rem 0;
    }

    .stSlider {
        margin: 0.2rem 0;
    }
    
    .stSlider > div > div > div > div {
        font-size: 0.8rem;
    }
    
    .stCheckbox {
        margin: 0.2rem 0;
    }
    
    .element-container {
        margin: 0.1rem 0;
    }
    
    .main {
        height: 100vh;
        overflow: hidden;
    }
    
    .stHorizontalBlock {
        height: calc(100vh - 4rem);
        align-items: stretch;
    }
    
    .control-panel {
        height: calc(100vh - 4rem);
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    
    .plot-container {
        height: calc(100vh - 4rem);
        display: flex;
        align-items: center;
        justify-content: center;
    }
</style>

<script>
function adjustLayout() {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
    
    window.dispatchEvent(new Event('resize'));
}

window.addEventListener('load', adjustLayout);
window.addEventListener('resize', adjustLayout);
adjustLayout();
</script>
""", unsafe_allow_html=True)

st.title("Analytical SOP Simulator")

col1, col2 = st.columns([1, .6])

with col1:
    
    st.header("Parameters")
    
    col1a, col1b = st.columns(2)
    with col1a:
        onset = st.slider("Onset", 0.0, 100.0, 10.0, step=0.1, key="onset")
    with col1b:

        duration_0 = 50.0 if onset <= 49 else 99-onset
        
        T = st.slider("Duration", 0.0, 100.0-onset, duration_0, step=0.1, key="duration")

    
    col2a, col2b, col2c = st.columns(3)
    with col2a:
        p1 = st.slider("p1", 0.0, 0.99, 0.2, step=0.01, key="p1")
    with col2b:
        pd1 = st.slider("pd1", 0.01, 0.99, 0.1, step=0.01, key="pd1")
    with col2c:
        pd2 = st.slider("pd2", 0.01, 0.99, 0.02, step=0.01, key="pd2")
    
    assoc = st.checkbox("Associative influences?", key="assoc")
    if assoc:
        p2 = st.slider("p2", 0.0, 0.99, 0.1, step=0.01, key="p2")
    else:
        p2 = 0.0

total_time = 100.0

class SOP_analytical:
    def __init__(self, t, T, p1=0.9, p2=0.1, pd1=0.1, pd2=0.02, onset=0.0):

        self.t = np.asarray(t)
        self.T = T
        self.p1, self.p2 = p1, p2
        self.pd1, self.pd2 = pd1, pd2
        self.onset = onset

        self.omega = (p1 + p2 + pd1 + pd2) / 2
        self.D = p1*pd1 + p1*pd2 + p2*pd1 + pd1*pd2
        self.theta = self.omega**2 - self.D
        self.sqrt_theta = np.sqrt(abs(self.theta))

        self.idx_start = np.searchsorted(self.t, onset)
        self.idx_end = np.searchsorted(self.t, onset + T)

        self.t_on = self.t[self.idx_start:self.idx_end] - onset  # 0 to T
        self.t_off = self.t[self.idx_end:] - (onset + T)        # > T

        self.exp_term = np.exp(-self.omega * self.t_on)
        if self.theta > 0:
            self.cosh_term = np.cosh(self.sqrt_theta * self.t_on)
            self.sinh_term = np.sinh(self.sqrt_theta * self.t_on) / self.sqrt_theta
        elif self.theta < 0:
            self.cosh_term = np.cos(self.sqrt_theta * self.t_on)
            self.sinh_term = np.sin(self.sqrt_theta * self.t_on) / self.sqrt_theta
        else:
            self.cosh_term = np.ones_like(self.t_on)
            self.sinh_term = self.t_on

    def A1_t(self):
        A1 = np.zeros_like(self.t)
        if self.idx_start < self.idx_end:
            num = self.p1 * self.pd2
            factor = (self.D/self.pd2) - self.omega
            A1_on = ((num / self.D)
                     - (num * self.exp_term * (self.cosh_term - self.sinh_term * factor)) / self.D)
            last_val = A1_on[-1]
            A1_off = last_val * np.exp(-self.pd1 * self.t_off)
            A1[self.idx_start:self.idx_end] = A1_on
            A1[self.idx_end:] = A1_off
        return A1

    def A2_t(self):
        A2 = np.zeros_like(self.t)
        if self.idx_start < self.idx_end:
            num = self.pd1 * (self.p1 + self.p2)
            sum_term = num
            sinh_factor = (self.omega - self.D/self.pd2
                           + (self.pd1*(self.p1 + self.p2))/self.pd2
                           - self.p1*self.pd2*(self.pd1 - self.p2)
                             /(self.pd1*(self.p1 + self.p2)))
            A2_on = (num/self.D)
            A2_on -= (self.exp_term * (self.cosh_term - self.sinh_term * sinh_factor) * sum_term)/self.D

            t_rel = self.t_off
            exp_p = np.exp(-(self.p2 + self.pd2) * t_rel)
            exp_d1 = np.exp(-self.pd1 * t_rel)
            val_on_end = A2_on[-1]
            val_A1_end = self.A1_t()[self.idx_end-1]

            if abs((self.p2 + self.pd2) - self.pd1) > 1e-12:
                dec = (val_on_end * exp_p
                       + self.p2/(self.p2+self.pd2) * (1 - exp_p)
                       + val_A1_end*(self.pd1 - self.p2)
                         /(self.p2 + self.pd2 - self.pd1) * (exp_d1 - exp_p))
            else:
                dec = (val_on_end * exp_p
                       + self.p2/(self.p2+self.pd2) * (1 - exp_p)
                       + val_A1_end * (self.pd1 - self.p2) * t_rel * exp_d1)
            A2[self.idx_start:self.idx_end] = A2_on
            A2[self.idx_end:] = dec
        return A2

    def I_t(self):
        return 1.0 - self.A1_t() - self.A2_t()

t = np.arange(0, 100.1, .01)
model = SOP_analytical(t, T, p1, p2, pd1, pd2, onset)
a1 = model.A1_t()
a2 = model.A2_t()
i = model.I_t()

with col2:
    plt.rcParams.update({
    'text.usetex': True,        
    'font.family': 'serif',     
    'axes.facecolor': 'white',   
    'figure.facecolor': 'white', 
    'text.color': 'black',       
    'axes.edgecolor': 'black',   
    'axes.labelcolor': 'black',  
    'xtick.color': 'black',      
    'ytick.color': 'black',      
    })

    time = np.arange(0,100.1,.1)
    pulse = np.where((time >= onset) & (time < onset+T), 1, 0)
    fig = plt.figure(figsize=(4,4.5), dpi=200,constrained_layout=True)
    ax = plt.subplot2grid((60, 1), (0, 0), rowspan=50)
    ax.plot(t, i,  label=r'$I$', linestyle=":", color="black")
    ax.plot(t, a1, label=r'$A1$',linestyle="-", color="black")
    ax.plot(t, a2, label=r'$A2$',linestyle="--", color="black")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_ylabel(r'Proportion of activity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=13, loc="upper right")
    ax.legend()

    ax2 = plt.subplot2grid((60, 1), (50, 0), rowspan=5)
    ax2.fill_between(time, 0, pulse*0.12, color='black', step='post')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, .2)
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.set_yticks([])
    ax2.set_xlabel("Step time", fontsize=13)
    ax2.tick_params(labelsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    st.pyplot(fig, use_container_width=True, clear_figure=True)
    
    st.markdown('</div>', unsafe_allow_html=True)