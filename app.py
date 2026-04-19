import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import json
import glob
from collections import deque
from game import SnakeGameAI, BLOCK_SIZE
from agent import Agent

st.set_page_config(page_title="Deep Q-Learning Snake", layout="wide", page_icon="🐍", initial_sidebar_state="expanded")

# Theme selector
theme = st.sidebar.radio("UI Theme", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0f172a"
    grid_color = "rgba(255,255,255,0.05)"
    snake_head = "#38bdf8"
    snake_body = "#0284c7"
    food_color = "#f43f5e"
    eye_color = "#0f172a"
    template = "plotly_dark"
    st.markdown("""
    <style>
        .stApp { background-color: #020617; color: #f8fafc; }
        .stMetric { background-color: #0f172a; padding: 15px; border-radius: 10px; border: 1px solid #1e293b; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 800; color: #38bdf8; }
        .stButton>button { border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
else:
    bg_color = "#f8fafc"
    grid_color = "rgba(0,0,0,0.05)"
    snake_head = "#0ea5e9"
    snake_body = "#38bdf8"
    food_color = "#e11d48"
    eye_color = "#ffffff"
    template = "plotly_white"
    st.markdown("""
    <style>
        .stApp { background-color: #f1f5f9; color: #0f172a; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
        div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 800; color: #0ea5e9; }
        .stButton>button { border-radius: 8px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def draw_snake_svg(game):
    svg = f'<div style="display:flex;justify-content:center;margin-top:20px;"><svg width="{game.w}" height="{game.h}" xmlns="http://www.w3.org/2000/svg" style="background-color: {bg_color}; border-radius: 12px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2); border: 2px solid {snake_body};">'
    
    # Grid
    for x in range(0, game.w, BLOCK_SIZE):
        svg += f'<line x1="{x}" y1="0" x2="{x}" y2="{game.h}" stroke="{grid_color}" stroke-width="1" />'
    for y in range(0, game.h, BLOCK_SIZE):
        svg += f'<line x1="0" y1="{y}" x2="{game.w}" y2="{y}" stroke="{grid_color}" stroke-width="1" />'

    # Food
    svg += f'<circle cx="{game.food.x + BLOCK_SIZE/2}" cy="{game.food.y + BLOCK_SIZE/2}" r="{BLOCK_SIZE/2 - 2}" fill="{food_color}" filter="drop-shadow(0px 0px 5px {food_color})"/>'

    # Snake
    for i, pt in enumerate(game.snake):
        color = snake_head if i == 0 else snake_body
        svg += f'<rect x="{pt.x+1}" y="{pt.y+1}" width="{BLOCK_SIZE-2}" height="{BLOCK_SIZE-2}" rx="6" fill="{color}" />'
        if i == 0:
            svg += f'<circle cx="{pt.x + 6}" cy="{pt.y + 6}" r="2" fill="{eye_color}" />'
            svg += f'<circle cx="{pt.x + 14}" cy="{pt.y + 6}" r="2" fill="{eye_color}" />'
            
    svg += '</svg></div>'
    return svg

def get_models():
    models = []
    if os.path.exists('./models'):
        for file in glob.glob('./models/*.json'):
            try:
                with open(file, 'r') as f:
                    meta = json.load(f)
                    meta['file'] = file.replace('.json', '.pth')
                    models.append(meta)
            except: pass
    return sorted(models, key=lambda x: x.get('best_score', 0), reverse=True)

# Initialize Session State
if "history" not in st.session_state: st.session_state.history = []
if "is_training" not in st.session_state: st.session_state.is_training = False
if "is_simulating" not in st.session_state: st.session_state.is_simulating = False

# Sidebar Navigation
st.sidebar.title("🐍 Navigation")
nav = st.sidebar.radio("Go to:", ["🏠 Home Overview", "🏋️ Train Agent", "📈 Analytics", "🎮 Watch AI Play", "💾 Model Manager"])
st.sidebar.markdown("---")
st.sidebar.info("This project demonstrates a Deep Q-Network learning to play Snake autonomously.")

# =======================
# HOME TAB
# =======================
if nav == "🏠 Home Overview":
    st.title("Snake Automation via Deep Q-Learning")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        ### Welcome to the Interactive RL Dashboard!
        This web application allows you to train, evaluate, and visualize a **Deep Q-Network (DQN)** agent learning to play Snake.
        
        #### 🧠 How it Works:
        1. **State:** The snake sees its immediate surroundings (11 values: danger left/straight/right, current direction, food location).
        2. **Action:** The Neural Network outputs 3 Q-Values (go straight, turn left, turn right).
        3. **Reward:** It receives `+10` for food, `-10` for death, and small shaped rewards to encourage moving towards food.
        4. **Memory:** It stores past experiences in a Replay Buffer and learns from randomized mini-batches.
        """)
    with col2:
        st.info("""
        ### Workflow:
        1. **Train** an agent in the *Train Agent* tab.
        2. Check its progress in *Analytics*.
        3. **Simulate** the model in *Watch AI Play*.
        4. Manage your models in *Model Manager*.
        """)
        
# =======================
# TRAIN AGENT TAB
# =======================
elif nav == "🏋️ Train Agent":
    st.title("Train DQN Agent")
    st.markdown("---")
    
    c1, c2 = st.columns([1, 3])
    
    with c1:
        with st.expander("⚙️ Hyperparameters", expanded=True):
            params = {}
            params['episodes'] = st.number_input("Target Episodes", min_value=1, value=100)
            params['learning_rate'] = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f")
            params['gamma'] = st.slider("Gamma (Discount factor)", 0.0, 1.0, 0.9, step=0.05)
            params['epsilon_start'] = st.slider("Epsilon Start", 0, 200, 80)
            params['epsilon_min'] = st.slider("Epsilon Min", 0, 50, 0)
            params['epsilon_decay'] = st.number_input("Epsilon Decay", value=1)
            params['batch_size'] = st.number_input("Batch Size", value=1000)
            params['max_steps'] = st.number_input("Max Steps per Ep", value=2000)
            
        train_btn = st.button("🚀 Start Training", type="primary", use_container_width=True)
        stop_btn = st.button("🛑 Stop Training", use_container_width=True)
        
    with c2:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        m_eps = metric_col1.empty()
        m_score = metric_col2.empty()
        m_best = metric_col3.empty()
        m_avg = metric_col4.empty()
        
        # Initialize default metrics
        m_eps.metric("Episode", "0 / 0")
        m_score.metric("Last Score", "0")
        m_best.metric("Best Score", "0")
        m_avg.metric("Avg Score", "0.00")
        
        svg_ph = st.empty()
        chart_ph = st.empty()
        
        if stop_btn:
            st.session_state.is_training = False
            
        if train_btn:
            st.session_state.is_training = True
            st.session_state.history = []
            agent = Agent(params)
            game = SnakeGameAI(max_steps=params['max_steps'])
            record = 0
            total_score = 0
            
            progress_bar = st.progress(0)
            best_weights = None
            best_moving_avg = 0.0
            recent_scores = deque(maxlen=20)
            
            for ep in range(1, params['episodes'] + 1):
                if not st.session_state.is_training:
                    st.warning("Training stopped by user.")
                    break
                    
                game.reset()
                while True:
                    state_old = agent.get_state(game)
                    final_move = agent.get_action(state_old, evaluate=False)
                    state_new, reward, done, score = game.play_step(final_move)
                    
                    agent.train_short_memory(state_old, final_move, reward, state_new, done)
                    agent.remember(state_old, final_move, reward, state_new, done)
                    
                    if done:
                        agent.n_games += 1
                        loss = agent.train_long_memory()
                        if score > record: 
                            record = score
                            
                        recent_scores.append(score)
                        if len(recent_scores) >= 10:
                            current_m_avg = sum(recent_scores) / len(recent_scores)
                            if current_m_avg > best_moving_avg:
                                best_moving_avg = current_m_avg
                                import copy
                                best_weights = copy.deepcopy(agent.model.state_dict())
                            
                        total_score += score
                        avg = total_score / agent.n_games
                        
                        st.session_state.history.append({
                            'episode': agent.n_games, 'score': score, 'average': avg,
                            'epsilon': agent.epsilon, 'loss': loss
                        })
                        
                        # UI Updates
                        if ep % 5 == 0 or ep == 1 or ep == params['episodes']:
                            m_eps.metric("Episode", f"{agent.n_games} / {params['episodes']}")
                            m_score.metric("Last Score", score)
                            m_best.metric("Best Score", record)
                            m_avg.metric("Avg Score", f"{avg:.2f}")
                            
                            svg_ph.info("⚡ Visual simulation is hidden during training for maximum speed. You can watch the AI play in the 'Watch AI Play' tab after training is complete!")
                            
                            df = pd.DataFrame(st.session_state.history)
                            fig = px.line(df, x='episode', y=['score', 'average'], title="Live Training Performance", template=template)
                            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300)
                            chart_ph.plotly_chart(fig, use_container_width=True)
                            
                            progress_bar.progress(ep / params['episodes'])
                        break
                        
            if st.session_state.is_training:
                st.success("Training Complete!")
                if best_weights is not None:
                    agent.model.load_state_dict(best_weights)
                agent.save_model("model", record, total_score/agent.n_games)
                st.session_state.is_training = False

# =======================
# ANALYTICS TAB
# =======================
elif nav == "📈 Analytics":
    st.title("Training Analytics")
    st.markdown("---")
    
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.scatter(df, x='episode', y='score', title="Score vs Episodes (Growth)", template=template)
            fig1.update_traces(marker=dict(color="#38bdf8", opacity=0.6))
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            fig2 = px.line(df, x='episode', y='average', title="Moving Average Score", template=template)
            fig2.update_traces(line_color="#10b981", line_width=3)
            st.plotly_chart(fig2, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.line(df, x='episode', y='epsilon', title="Exploration Rate (Epsilon) Decay", template=template)
            fig3.update_traces(line_color="#f59e0b", line_width=3)
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            fig4 = px.line(df, x='episode', y='loss', title="Neural Network Loss", template=template)
            fig4.update_traces(line_color="#ef4444")
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No training data available. Go to the Train Agent tab and run a session first!")

# =======================
# SIMULATION TAB
# =======================
elif nav == "🎮 Watch AI Play":
    st.title("Watch AI Play (Simulation)")
    st.markdown("---")
    
    models = get_models()
    
    if len(models) == 0:
        st.warning("No saved models found. Please train a model first.")
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            st.subheader("Simulation Controls")
            model_options = {m['file']: f"{m['name']} ({m['episodes']} ep) - Avg: {m['average_score']:.1f}" for m in models}
            selected_file = st.selectbox("Select Trained Model", options=list(model_options.keys()), format_func=lambda x: model_options[x])
            delay = st.slider("Simulation Speed (ms delay)", 10, 200, 80)
            
            st.markdown("<br>", unsafe_allow_html=True)
            sim_btn = st.button("▶️ Start Simulation", type="primary", use_container_width=True)
            sim_stop = st.button("⏹️ Stop Simulation", use_container_width=True)
            
            st.markdown("---")
            sim_score = st.empty()
            sim_step = st.empty()
            sim_score.metric("Current Score", "0")
            sim_step.metric("Survival Steps", "0")
            
        with c2:
            sim_svg_ph = st.empty()
            sim_svg_ph.info("Click 'Start Simulation' to watch the selected AI model play.")
            
        if sim_stop:
            st.session_state.is_simulating = False
            
        if sim_btn:
            st.session_state.is_simulating = True
            agent = Agent()
            agent.load_model(selected_file)
            game = SnakeGameAI()
            
            step = 0
            while st.session_state.is_simulating:
                state = agent.get_state(game)
                action = agent.get_action(state, evaluate=True)
                _, _, done, score = game.play_step(action)
                step += 1
                
                sim_score.metric("Current Score", score)
                sim_step.metric("Survival Steps", step)
                sim_svg_ph.markdown(draw_snake_svg(game), unsafe_allow_html=True)
                
                time.sleep(delay / 1000.0)
                
                if done:
                    st.toast(f"Game Over! Final Score: {score}")
                    game.reset()
                    step = 0

# =======================
# MODEL MANAGER TAB
# =======================
elif nav == "💾 Model Manager":
    st.title("Model Manager")
    st.markdown("---")
    
    models = get_models()
    if len(models) == 0:
        st.info("No models currently saved. Train an agent to populate this list.")
    else:
        df_models = pd.DataFrame(models)
        cols = ['name', 'episodes', 'best_score', 'average_score', 'date', 'file']
        df_models = df_models[[c for c in cols if c in df_models.columns]]
        
        # Rename columns to clarify these are training metrics, not evaluation metrics
        df_models = df_models.rename(columns={
            'name': 'Model Name',
            'episodes': 'Episodes Trained',
            'best_score': 'Training Best Score',
            'average_score': 'Training Avg Score',
            'date': 'Training Date',
            'file': 'File Path'
        })
        
        st.markdown("### Saved Trained Models")
        st.dataframe(df_models, use_container_width=True, hide_index=True)
        
        st.markdown("### Detailed Hyperparameters")
        selected_meta = st.selectbox("View Hyperparameters for Model", df_models['File Path'].tolist(), format_func=lambda x: [m['name'] for m in models if m['file']==x][0])
        meta = [m for m in models if m['file'] == selected_meta][0]
        st.json(meta.get('hyperparameters', {}))
