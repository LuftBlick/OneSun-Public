"""
Training & Monitoring Page
Start training and monitor progress in real-time
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import time
import json
from pathlib import Path

# Add src to path for imports
# Path structure: apps/main_platform/pages/03_... -> project_root
project_root = Path(__file__).resolve().parent.parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from utils.train import setup_training, train
from utils.model_eval import ModelEvaluator
from utils.loss_functions import combined_loss
from utils.colors import apply_theme, get_color, colored_metric, status_badge

# Apply OneSun theme
apply_theme()

st.title("🚀 Training & Monitoring")
st.markdown("Start model training and monitor progress in real-time.")

# Get references from session state
training_config = st.session_state.training_config
dataset_manager = st.session_state.dataset_manager

# Check if we have a valid configuration
if not training_config or not training_config.get('dataset_name'):
    st.markdown(f"""
    <div style="
        background-color: {get_color('error')}20;
        border: 1px solid {get_color('error')};
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        color: {get_color('error')};
        font-weight: 600;
        margin: 2rem 0;
    ">
        ❌ No training configuration found!<br>
        <span style="font-weight: 400; margin-top: 0.5rem; display: block;">
            Please configure your training in Training Configuration first.
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Display current configuration summary
st.header("📋 Current Configuration")

# Create styled configuration display
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {get_color('primary')}20 0%, {get_color('primary')}10 100%);
        border: 1px solid {get_color('primary')}40;
        border-radius: 8px;
        padding: 1rem;
        height: 140px;
    ">
        <div style="
            color: {get_color('primary')};
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.75rem;
        ">📊 Dataset & Models</div>
        <div style="color: {get_color('text_primary')}; line-height: 1.6;">
            <strong>Dataset:</strong> {training_config['dataset_name']}<br>
            <strong>Instrument:</strong> {training_config.get('instrument_model', 'Unknown')}<br>
            <strong>Atmospheric:</strong> {training_config.get('atmospheric_model', 'Unknown')}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {get_color('secondary')}20 0%, {get_color('secondary')}10 100%);
        border: 1px solid {get_color('secondary')}40;
        border-radius: 8px;
        padding: 1rem;
        height: 140px;
    ">
        <div style="
            color: {get_color('secondary')};
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.75rem;
        ">🚀 Training Parameters</div>
        <div style="color: {get_color('text_primary')}; line-height: 1.6;">
            <strong>Epochs:</strong> {training_config.get('epochs', 'Unknown')}<br>
            <strong>Batch Size:</strong> {training_config.get('batch_size', 'Unknown')}<br>
            <strong>Learning Rate:</strong> {training_config.get('learning_rate', 'Unknown')}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Handle loss function name safely


    save_icon = "✅" if training_config.get('save', True) else "❌"
    early_stop_icon = "✅" if training_config.get('early_stopping', False) else "❌"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {get_color('accent')}20 0%, {get_color('accent')}10 100%);
        border: 1px solid {get_color('accent')}40;
        border-radius: 8px;
        padding: 1rem;
        height: 140px;
    ">
        <div style="
            color: {get_color('accent')};
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 0.75rem;
        ">⚙️ Settings</div>
        <div style="color: {get_color('text_primary')}; line-height: 1.6;">
            <strong>Save Models:</strong> {save_icon}<br>
            <strong>Early Stopping:</strong> {early_stop_icon}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Training section
if "training_status" not in st.session_state or st.session_state.training_status != "completed":
    st.header("🚀 Start Training")
    
    # Validate dataset exists
    dataset_exists = dataset_manager.load_dataset(training_config['dataset_name'], dataset_type='train') is not None
    
    if not dataset_exists:
        st.markdown(f"""
        <div style="
            background-color: {get_color('error')}20;
            border: 1px solid {get_color('error')};
            border-radius: 8px;
            padding: 1rem;
            color: {get_color('error')};
            font-weight: 600;
            text-align: center;
        ">
            ❌ Dataset '{training_config['dataset_name']}' not found!<br>
            <span style="font-weight: 400;">Please create the dataset first in the Dataset Management page.</span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Training controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**Ready to start training with the configuration above.**")
        if training_config.get('early_stopping'):
            st.markdown(f"""
            <div style="
                background-color: {get_color('info')}20;
                border-left: 4px solid {get_color('info')};
                padding: 0.75rem;
                margin: 0.5rem 0;
                border-radius: 0 6px 6px 0;
                color: {get_color('info')};
            ">
                ℹ️ Early stopping enabled with patience: {training_config.get('patience', 5)} epochs
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        start_training = st.button(
            "🚀 Start Training", 
            key="start_training_button",
            use_container_width=True,
            type="primary"
        )
    
    if start_training:
        # Initialize progress tracking
        progress_container = st.container()
        
        with progress_container:
            # Step 1: Setup phase
            setup_status = st.empty()
            setup_status.markdown(f"""
            <div style="
                background-color: {get_color('info')}20;
                border: 1px solid {get_color('info')};
                border-radius: 6px;
                padding: 1rem;
                color: {get_color('info')};
                font-weight: 600;
                text-align: center;
            ">
                🔧 Setting up training environment...
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Setup training components
                components = setup_training(training_config)
                
                # Store components for later use
                st.session_state.components = components
                setup_status.markdown(f"""
                <div style="
                    background-color: {get_color('success')}20;
                    border: 1px solid {get_color('success')};
                    border-radius: 6px;
                    padding: 1rem;
                    color: {get_color('success')};
                    font-weight: 600;
                    text-align: center;
                ">
                    ✅ Training environment set up successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Step 2: Training phase
                training_status = st.empty()
                training_status.markdown(f"""
                <div style="
                    background-color: {get_color('primary')}20;
                    border: 1px solid {get_color('primary')};
                    border-radius: 6px;
                    padding: 1rem;
                    color: {get_color('primary')};
                    font-weight: 600;
                    text-align: center;
                ">
                    🚀 Training in progress... This may take a while.
                </div>
                """, unsafe_allow_html=True)
                
                # Progress tracking
                progress_bar = st.progress(0)
                loss_chart_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                # Custom training function with real-time updates
                def train_with_monitoring(components):
                    """Training with real-time Streamlit updates"""
                    config = components['config']
                    trainer = components['trainer']
                    train_loader = components['train_loader']
                    val_loader = components['val_loader']
                    model_i = components['model_i']
                    model_a = components['model_a']
                    optimizer_i = components['optimizer_i']
                    optimizer_a = components['optimizer_a']
                    
                    # Create evaluator
                    evaluator = ModelEvaluator(trainer=trainer)
                    
                    # Training tracking
                    train_losses = []
                    valid_losses = []
                    early_stopped = False
                    start_time = time.time()
                    
                    try:
                        for epoch in range(config['epochs']):
                            epoch_start = time.time()
                            
                            # Update progress
                            progress = (epoch + 1) / config['epochs']
                            progress_bar.progress(progress)
                            training_status.markdown(f"""
                            <div style="
                                background-color: {get_color('primary')}20;
                                border: 1px solid {get_color('primary')};
                                border-radius: 6px;
                                padding: 1rem;
                                color: {get_color('primary')};
                                font-weight: 600;
                                text-align: center;
                            ">
                                🚀 Training Epoch {epoch+1}/{config['epochs']}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Train epoch
                            train_loss = trainer.train_epoch(
                                data_loader=train_loader,
                                optimizer_i=optimizer_i,
                                optimizer_a=optimizer_a,
                                epoch=epoch
                            )
                            train_losses.append(train_loss)
                            
                            # Validation
                            _, val_loss = evaluator.evaluate_models(
                                model_i, model_a, val_loader,
                             mse_loss_w=config['mse_loss_w'], corr_loss_w=config['corr_loss_w'], sam_loss_w=config['sam_loss_w'], cos_loss_w=config['cos_loss_w']
                            )
                            valid_losses.append(val_loss)
                            
                            # Update live metrics with OneSun styling
                            with metrics_placeholder.container():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f"""
                                    <div style="
                                        background-color: {get_color('primary')}20;
                                        border: 1px solid {get_color('primary')}40;
                                        border-radius: 6px;
                                        padding: 1rem;
                                        text-align: center;
                                    ">
                                        <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">Current Epoch</div>
                                        <div style="color: {get_color('primary')}; font-size: 1.5rem; font-weight: bold;">{epoch+1}/{config['epochs']}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f"""
                                    <div style="
                                        background-color: {get_color('secondary')}20;
                                        border: 1px solid {get_color('secondary')}40;
                                        border-radius: 6px;
                                        padding: 1rem;
                                        text-align: center;
                                    ">
                                        <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">Training Loss</div>
                                        <div style="color: {get_color('secondary')}; font-size: 1.5rem; font-weight: bold;">{train_loss:.6f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with col3:
                                    st.markdown(f"""
                                    <div style="
                                        background-color: {get_color('accent')}20;
                                        border: 1px solid {get_color('accent')}40;
                                        border-radius: 6px;
                                        padding: 1rem;
                                        text-align: center;
                                    ">
                                        <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">Validation Loss</div>
                                        <div style="color: {get_color('accent')}; font-size: 1.5rem; font-weight: bold;">{val_loss:.6f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Update live loss chart with OneSun colors
                            if len(train_losses) > 1:
                                epochs_range = list(range(1, len(train_losses) + 1))
                                loss_df = pd.DataFrame({
                                    'epoch': epochs_range + epochs_range,
                                    'loss': train_losses + valid_losses,
                                    'type': ['Training'] * len(epochs_range) + ['Validation'] * len(epochs_range)
                                })
                                
                                fig = px.line(
                                    loss_df, x='epoch', y='loss', color='type',
                                    title='🚀 Training Progress',
                                    labels={'epoch': 'Epoch', 'loss': 'Loss', 'type': 'Dataset'},
                                    color_discrete_map={
                                        'Training': get_color('primary'),
                                        'Validation': get_color('secondary')
                                    }
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(color=get_color('text_primary'))
                                )
                                loss_chart_placeholder.plotly_chart(fig, use_container_width=True)
                            
                            # Save checkpoint
                            if config.get('save', True):
                                checkpoint_data = {
                                    'epoch': epoch,
                                    'model_i_state_dict': model_i.state_dict(),
                                    'model_a_state_dict': model_a.state_dict(),
                                    'optimizer_i_state_dict': optimizer_i.state_dict(),
                                    'optimizer_a_state_dict': optimizer_a.state_dict(),
                                    'train_loss': train_loss,
                                    'val_loss': val_loss,
                                    'train_losses': train_losses,
                                    'valid_losses': valid_losses
                                }
                                trainer.save_checkpoint_data(checkpoint_data)
                            
                            # Early stopping check
                            if config.get('early_stopping', False) and len(valid_losses) > config.get('patience', 5):
                                patience = config.get('patience', 5)
                                if val_loss > min(valid_losses[:-patience]):
                                    training_status.markdown(f"""
                                    <div style="
                                        background-color: {get_color('warning')}20;
                                        border: 1px solid {get_color('warning')};
                                        border-radius: 6px;
                                        padding: 1rem;
                                        color: {get_color('warning')};
                                        font-weight: 600;
                                        text-align: center;
                                    ">
                                        🛑 Early stopping triggered: no improvement for {patience} epochs
                                    </div>
                                    """, unsafe_allow_html=True)
                                    early_stopped = True
                                    break
                    
                    except KeyboardInterrupt:
                        training_status.markdown(f"""
                        <div style="
                            background-color: {get_color('warning')}20;
                            border: 1px solid {get_color('warning')};
                            border-radius: 6px;
                            padding: 1rem;
                            color: {get_color('warning')};
                            font-weight: 600;
                            text-align: center;
                        ">
                            ⚠️ Training interrupted by user
                        </div>
                        """, unsafe_allow_html=True)
                    
                    finally:
                        # Calculate final metrics
                        training_time = time.time() - start_time
                        
                        # Save final models
                        if config.get('save', True):
                            trainer.save_final_models()
                            
                            # Save comprehensive metrics
                            metrics_path = trainer.run_dir / "experiment_metrics.json"
                            metrics = {
                                'train_losses': train_losses,
                                'valid_losses': valid_losses,
                                'best_val_loss': min(valid_losses) if valid_losses else None,
                                'best_epoch': valid_losses.index(min(valid_losses)) + 1 if valid_losses else None,
                                'training_time': training_time,
                                'epochs_completed': len(train_losses),
                                'early_stopped': early_stopped,
                                'final_lr_i': optimizer_i.param_groups[0]['lr'],
                                'final_lr_a': optimizer_a.param_groups[0]['lr'] if optimizer_a is not None else None,
                                'mean_epoch_time': training_time / len(train_losses) if train_losses else 0,
                                'train_val_loss_ratio': train_losses[-1] / valid_losses[-1] if train_losses and valid_losses else None,
                                'instrument_model': config.get('instrument_model', 'Unknown'),
                                'atmospheric_model': config.get('atmospheric_model', 'Unknown'),
                                'gas_ods': config.get('gas_ods', 'Unknown'),
                                'solar_conv': config.get('solar_conv', 'Unknown')
                            }
                            with open(metrics_path, 'w') as f:
                                json.dump(metrics, f, indent=4, default=lambda o: float(o) if isinstance(o, np.float32) else o)
                        
                        # Final status update
                        progress_bar.progress(100)
                        training_status.markdown(f"""
                        <div style="
                            background-color: {get_color('success')}20;
                            border: 1px solid {get_color('success')};
                            border-radius: 6px;
                            padding: 1rem;
                            color: {get_color('success')};
                            font-weight: 600;
                            text-align: center;
                        ">
                            ✅ Training completed in {training_time:.2f} seconds!
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Store results in session state
                        results = {
                            'train_losses': train_losses,
                            'valid_losses': valid_losses,
                            'training_time': training_time,
                            'early_stopped': early_stopped,
                            'components': components
                        }
                        
                        st.session_state.training_results = results
                        st.session_state.train_losses = train_losses
                        st.session_state.valid_losses = valid_losses
                        st.session_state.training_time = training_time
                        st.session_state.early_stopped = early_stopped
                        st.session_state.components = components
                        st.session_state.training_status = "completed"
                        
                        trainer.close()
                        
                        return results
                
                # Run training with monitoring
                results = train_with_monitoring(components)
                
                # Auto-refresh to show results section
                st.rerun()
                
            except Exception as e:
                setup_status.markdown(f"""
                <div style="
                    background-color: {get_color('error')}20;
                    border: 1px solid {get_color('error')};
                    border-radius: 6px;
                    padding: 1rem;
                    color: {get_color('error')};
                    font-weight: 600;
                    text-align: center;
                ">
                    ❌ Error during training: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.exception(e)
                st.session_state.training_status = "error"

# Results section
elif "training_status" in st.session_state and st.session_state.training_status == "completed":
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {get_color('success')} 0%, {get_color('primary')} 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    ">
        🎉 Training Complete!
    </div>
    """, unsafe_allow_html=True)
    
    # Training summary with OneSun styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="
            background-color: {get_color('primary')}20;
            border: 1px solid {get_color('primary')}40;
            border-radius: 6px;
            padding: 1rem;
            text-align: center;
        ">
            <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">Training Time</div>
            <div style="color: {get_color('primary')}; font-size: 1.5rem; font-weight: bold;">{st.session_state.training_time:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        best_epoch = np.argmin(st.session_state.valid_losses) + 1
        st.markdown(f"""
        <div style="
            background-color: {get_color('secondary')}20;
            border: 1px solid {get_color('secondary')}40;
            border-radius: 6px;
            padding: 1rem;
            text-align: center;
        ">
            <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">Best Epoch</div>
            <div style="color: {get_color('secondary')}; font-size: 1.5rem; font-weight: bold;">{best_epoch}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_val_loss = min(st.session_state.valid_losses)
        st.markdown(f"""
        <div style="
            background-color: {get_color('accent')}20;
            border: 1px solid {get_color('accent')}40;
            border-radius: 6px;
            padding: 1rem;
            text-align: center;
        ">
            <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">Best Val Loss</div>
            <div style="color: {get_color('accent')}; font-size: 1.5rem; font-weight: bold;">{best_val_loss:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        final_train_loss = st.session_state.train_losses[-1]
        st.markdown(f"""
        <div style="
            background-color: {get_color('success')}20;
            border: 1px solid {get_color('success')}40;
            border-radius: 6px;
            padding: 1rem;
            text-align: center;
        ">
            <div style="color: {get_color('text_secondary')}; font-size: 0.9rem;">Final Train Loss</div>
            <div style="color: {get_color('success')}; font-size: 1.5rem; font-weight: bold;">{final_train_loss:.6f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Early stopping info
    if st.session_state.early_stopped:
        st.markdown(f"""
        <div style="
            background-color: {get_color('info')}20;
            border-left: 4px solid {get_color('info')};
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 6px 6px 0;
            color: {get_color('info')};
        ">
            ℹ️ Training was stopped early due to no improvement in validation loss.
        </div>
        """, unsafe_allow_html=True)
    
    # Loss curves visualization
    st.subheader("📈 Training Progress")
    
    epochs = list(range(1, len(st.session_state.train_losses) + 1))
    loss_df = pd.DataFrame({
        'epoch': epochs + epochs,
        'loss': st.session_state.train_losses + st.session_state.valid_losses,
        'type': ['Training'] * len(epochs) + ['Validation'] * len(epochs)
    })
    
    fig = px.line(
        loss_df, x='epoch', y='loss', color='type',
        markers=True,
        title='Training vs. Validation Loss',
        labels={'epoch': 'Epoch', 'loss': 'Loss', 'type': 'Dataset'},
        color_discrete_map={
            'Training': get_color('primary'),
            'Validation': get_color('secondary')
        }
    )
    fig.update_layout(
        xaxis=dict(dtick=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=get_color('text_primary'))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Experiment artifacts management
    st.subheader("🗂️ Experiment Management")
    
    if 'components' in st.session_state and 'trainer' in st.session_state.components:
        run_dir = st.session_state.components['trainer'].run_dir
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="
                background-color: {get_color('info')}20;
                border: 1px solid {get_color('info')};
                border-radius: 6px;
                padding: 1rem;
                color: {get_color('info')};
                font-weight: 500;
            ">
                📁 <strong>Experiment saved to:</strong><br>
                <code>{run_dir}</code>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🗑️ Discard Run", key="discard_run"):
                import shutil
                try:
                    shutil.rmtree(run_dir)
                    st.markdown(f"""
                    <div style="
                        background-color: {get_color('success')}20;
                        border: 1px solid {get_color('success')};
                        border-radius: 6px;
                        padding: 1rem;
                        color: {get_color('success')};
                        font-weight: 600;
                        text-align: center;
                    ">
                        ✅ Discarded run directory: {run_dir.name}
                    </div>
                    """, unsafe_allow_html=True)
                    # Reset training status
                    for key in ["training_status", "training_results", "train_losses", "valid_losses"]:
                        if key in st.session_state:
                            del st.session_state[key]
                except Exception as e:
                    st.markdown(f"""
                    <div style="
                        background-color: {get_color('error')}20;
                        border: 1px solid {get_color('error')};
                        border-radius: 6px;
                        padding: 1rem;
                        color: {get_color('error')};
                        font-weight: 600;
                        text-align: center;
                    ">
                        ❌ Could not discard artifacts: {e}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Model architecture details
    with st.expander("🏗️ Model Architecture", expanded=False):
        if "components" in st.session_state:
            components = st.session_state.components
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔧 Instrument Model**")
                st.code(str(components['model_i']), language="text")
            
            with col2:
                st.markdown("**🌍 Atmospheric Model**")
                st.code(str(components['model_a']), language="text")
    
    # Next steps
    st.subheader("🎯 Next Steps")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {get_color('success')}20 0%, {get_color('primary')}20 100%);
            border: 1px solid {get_color('success')};
            border-radius: 8px;
            padding: 1.5rem;
            text-align: center;
            color: {get_color('success')};
            font-weight: 600;
        ">
            🔍 <strong>Analyze Results</strong><br>
            <span style="font-weight: 400; margin-top: 0.5rem; display: block;">
                Go to Analysis to explore your trained models!
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Start New Training", key="new_training"):
            # Reset training status to allow new training
            for key in ["training_status", "training_results", "train_losses", "valid_losses"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Sidebar information
with st.sidebar:
    st.markdown(f"""
    <div style="color: {get_color('text_primary')}; font-weight: 600; margin-bottom: 1rem;">
    🚀 Training Status
    </div>
    """, unsafe_allow_html=True)
    
    # Show training status
    if "training_status" in st.session_state:
        status = st.session_state.training_status
        if status == "completed":
            st.markdown(f"""
            <div style="
                background-color: {get_color('success')}20;
                color: {get_color('success')};
                padding: 0.75rem;
                border-radius: 6px;
                font-weight: 600;
                text-align: center;
                margin: 0.5rem 0;
                border: 1px solid {get_color('success')};
            ">✅ Training Complete</div>
            """, unsafe_allow_html=True)
            if "training_time" in st.session_state:
                st.markdown(f"⏱️ **Time:** {st.session_state.training_time:.1f}s")
            if "train_losses" in st.session_state:
                st.markdown(f"📊 **Epochs:** {len(st.session_state.train_losses)}")
        elif status == "running":
            st.markdown(f"""
            <div style="
                background-color: {get_color('info')}20;
                color: {get_color('info')};
                padding: 0.75rem;
                border-radius: 6px;
                font-weight: 600;
                text-align: center;
                margin: 0.5rem 0;
                border: 1px solid {get_color('info')};
            ">🔄 Training in Progress</div>
            """, unsafe_allow_html=True)
        elif status == "error":
            st.markdown(f"""
            <div style="
                background-color: {get_color('error')}20;
                color: {get_color('error')};
                padding: 0.75rem;
                border-radius: 6px;
                font-weight: 600;
                text-align: center;
                margin: 0.5rem 0;
                border: 1px solid {get_color('error')};
            ">❌ Training Error</div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="
            background-color: {get_color('text_muted')}20;
            color: {get_color('text_muted')};
            padding: 0.75rem;
            border-radius: 6px;
            font-weight: 600;
            text-align: center;
            margin: 0.5rem 0;
            border: 1px solid {get_color('text_muted')};
        ">⏳ Ready to Train</div>
        """, unsafe_allow_html=True)
    
    # Show experiment directory if available
    if 'components' in st.session_state and 'trainer' in st.session_state.components:
        run_dir = st.session_state.components['trainer'].run_dir
        st.markdown(f"📁 **Experiment:** `{run_dir.name}`")