#!/usr/bin/env python
"""
Generate an interactive Plotly HTML report from AOPC evaluation results.
"""
import argparse
import json
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    parser = argparse.ArgumentParser(description="Generate AOPC Plotly HTML report.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to aopc_evaluation_results.json")
    parser.add_argument("--out_html", type=str, default="aopc_visualization_report.html", help="Output HTML file path")
    args = parser.parse_args()

    with open(args.results_file, "r") as f:
        results = json.load(f)

    if not results:
        print("No results found in the JSON file.")
        return
        
    experiment_name = results[0].get("experiment", "Unknown Experiment")
    num_runs = len(results)
    run_name_text = f"Averaged over {num_runs} runs" if num_runs > 1 else results[0].get("run", "Unknown Run")
    
    # Extract all curves
    morf_curves = [r.get("morf_curve", []) for r in results if "morf_curve" in r]
    rand_curves = [r.get("rand_curve", []) for r in results if "rand_curve" in r]
    global_curves = [r.get("global_curve", []) for r in results if "global_curve" in r]
    gxi_morf_curves = [r.get("gxi_morf_curve", []) for r in results if "gxi_morf_curve" in r]
    gxi_global_curves = [r.get("gxi_global_curve", []) for r in results if "gxi_global_curve" in r]
    
    # Compute Means
    morf_mean = np.mean(morf_curves, axis=0).tolist() if morf_curves else []
    rand_mean = np.mean(rand_curves, axis=0).tolist() if rand_curves else []
    global_mean = np.mean(global_curves, axis=0).tolist() if global_curves else []
    gxi_morf_mean = np.mean(gxi_morf_curves, axis=0).tolist() if gxi_morf_curves else []
    gxi_global_mean = np.mean(gxi_global_curves, axis=0).tolist() if gxi_global_curves else []
    
    # Compute Stds
    morf_std = np.std(morf_curves, axis=0).tolist() if morf_curves else []
    rand_std = np.std(rand_curves, axis=0).tolist() if rand_curves else []
    global_std = np.std(global_curves, axis=0).tolist() if global_curves else []
    gxi_morf_std = np.std(gxi_morf_curves, axis=0).tolist() if gxi_morf_curves else []
    gxi_global_std = np.std(gxi_global_curves, axis=0).tolist() if gxi_global_curves else []
    # 1. Dataset-Wide AOPC Curve
    steps = list(range(1, len(morf_mean) + 1))
    
    fig_aopc = go.Figure()
    
    def add_shaded_curve(fig, x, y_mean, y_std, curves_list, name, color, fillcolor):
        # Mean line
        fig.add_trace(go.Scatter(x=x, y=y_mean, mode='lines', name=f'{name} Mean', line=dict(color=color, width=3), legendgroup=name))
        
        # If there's multiple runs, add individual runs
        if num_runs > 1:
            # Add individual run traces (faint)
            for i, curve in enumerate(curves_list):
                if curve:
                    fig.add_trace(go.Scatter(
                        x=x, y=curve,
                        mode='lines',
                        line=dict(color=color, width=1, dash='dot'),
                        opacity=0.3,
                        name=f'{name} Run {i}',
                        legendgroup=name,
                        showlegend=False,
                        hoverinfo="skip"
                    ))
            
    add_shaded_curve(fig_aopc, steps, morf_mean, morf_std, morf_curves, 'MORF AOPC', 'red', 'rgba(255,0,0,0.2)')
    add_shaded_curve(fig_aopc, steps, rand_mean, rand_std, rand_curves, 'Random AOPC', 'green', 'rgba(0,255,0,0.2)')
    add_shaded_curve(fig_aopc, steps, global_mean, global_std, global_curves, 'Global AOPC', 'blue', 'rgba(0,0,255,0.2)')
    add_shaded_curve(fig_aopc, steps, gxi_morf_mean, gxi_morf_std, gxi_morf_curves, 'GXI MORF AOPC', 'orange', 'rgba(255,165,0,0.2)')
    add_shaded_curve(fig_aopc, steps, gxi_global_mean, gxi_global_std, gxi_global_curves, 'GXI Global AOPC', 'purple', 'rgba(128,0,128,0.2)')
    
    fig_aopc.update_layout(
        title=f"Dataset-Wide AOPC Curve - {experiment_name} ({run_name_text})",
        xaxis_title="Perturbation Steps",
        yaxis_title="AOPC Score (Cumulative logit drop / K)",
        hovermode="x unified"
    )

    # 2. Example Logit Trajectories
    # We will plot up to 4 examples PER RUN.
    max_examples_per_run = 4
    all_subplot_titles = []
    
    # Calculate how many rows we need and build subplot titles
    total_examples_to_plot = 0
    run_example_counts = []
    
    for r_idx, run_data in enumerate(results):
        r_name = run_data.get("run", f"Run {r_idx}")
        m_logits = run_data.get("morf_logit_examples", [])
        n_ex = min(len(m_logits), max_examples_per_run)
        run_example_counts.append(n_ex)
        total_examples_to_plot += n_ex
        for i in range(n_ex):
            all_subplot_titles.append(f"[{r_name}] Sample {i+1} Target Class Logit")
            
    if total_examples_to_plot > 0:
        total_rows = (total_examples_to_plot + 1) // 2
        
        fig_logits = make_subplots(
            rows=total_rows, cols=2, 
            subplot_titles=all_subplot_titles,
            vertical_spacing=0.05
        )
        
        current_plot_idx = 0
        for r_idx, run_data in enumerate(results):
            morf_logits = run_data.get("morf_logit_examples", [])
            rand_logits = run_data.get("rand_logit_examples", [])
            global_logits = run_data.get("global_logit_examples", [])
            gxi_morf_logits = run_data.get("gxi_morf_logit_examples", [])
            gxi_global_logits = run_data.get("gxi_global_logit_examples", [])
            n_ex = run_example_counts[r_idx]
            
            if n_ex > 0:
                logit_steps = list(range(len(morf_logits[0])))
                
                def extract_1d(traj):
                    return [v[0] if isinstance(v, list) else v for v in traj]
                    
                for i in range(n_ex):
                    row = (current_plot_idx // 2) + 1
                    col = (current_plot_idx % 2) + 1
                    
                    show_leg = (current_plot_idx == 0)
                    fig_logits.add_trace(go.Scatter(x=logit_steps, y=extract_1d(morf_logits[i]), mode='lines', name=f'MORF', line=dict(color='red'), showlegend=show_leg, legendgroup='logit_morf'), row=row, col=col)
                    fig_logits.add_trace(go.Scatter(x=logit_steps, y=extract_1d(rand_logits[i]), mode='lines', name=f'Random', line=dict(color='green'), showlegend=show_leg, legendgroup='logit_rand'), row=row, col=col)
                    fig_logits.add_trace(go.Scatter(x=logit_steps, y=extract_1d(global_logits[i]), mode='lines', name=f'Global', line=dict(color='blue'), showlegend=show_leg, legendgroup='logit_glob'), row=row, col=col)
                    if gxi_morf_logits and i < len(gxi_morf_logits):
                        fig_logits.add_trace(go.Scatter(x=logit_steps, y=extract_1d(gxi_morf_logits[i]), mode='lines', name=f'GXI MORF', line=dict(color='orange'), showlegend=show_leg, legendgroup='logit_gxi_morf'), row=row, col=col)
                    if gxi_global_logits and i < len(gxi_global_logits):
                        fig_logits.add_trace(go.Scatter(x=logit_steps, y=extract_1d(gxi_global_logits[i]), mode='lines', name=f'GXI Global', line=dict(color='purple'), showlegend=show_leg, legendgroup='logit_gxi_glob'), row=row, col=col)
                    
                    fig_logits.update_xaxes(title_text="Perturbation Steps", row=row, col=col)
                    fig_logits.update_yaxes(title_text="Logit", row=row, col=col)
                    
                    current_plot_idx += 1
                    
        fig_logits.update_layout(
            title_text=f"Raw Logit Trajectories (Examples across all {num_runs} runs)",
            height=400 * total_rows, # Dynamically scale height
            hovermode="x unified"
        )
    else:
        fig_logits = go.Figure()
        
    # 3. Boxplot of Scalar Scores
    fig_box = go.Figure()
    if num_runs > 1:
        morf_scores = [r.get("morf_score", 0) for r in results if "morf_score" in r]
        rand_scores = [r.get("random_score", 0) for r in results if "random_score" in r]
        global_scores = [r.get("aopc_score", 0) for r in results if "aopc_score" in r]
        gxi_morf_scores = [r.get("gxi_morf_score", 0) for r in results if "gxi_morf_score" in r]
        gxi_global_scores = [r.get("gxi_global_score", 0) for r in results if "gxi_global_score" in r]
        
        fig_box.add_trace(go.Box(y=morf_scores, name='MORF AOPC Score', marker_color='red', boxpoints='all', jitter=0.3))
        fig_box.add_trace(go.Box(y=rand_scores, name='Random AOPC Score', marker_color='green', boxpoints='all', jitter=0.3))
        fig_box.add_trace(go.Box(y=global_scores, name='Global AOPC Score', marker_color='blue', boxpoints='all', jitter=0.3))
        fig_box.add_trace(go.Box(y=gxi_morf_scores, name='GXI MORF AOPC Score', marker_color='orange', boxpoints='all', jitter=0.3))
        fig_box.add_trace(go.Box(y=gxi_global_scores, name='GXI Global AOPC Score', marker_color='purple', boxpoints='all', jitter=0.3))
        
        fig_box.update_layout(
            title=f"Distribution of Scalar AOPC Scores ({num_runs} runs)",
            yaxis_title="Scalar Score",
            showlegend=False
        )
        
    # Write to a single HTML file
    
    # 4. Extract Orientation Stats
    orientation_html = ""
    orientation_stats_list = [r.get("orientation_stats") for r in results if r.get("orientation_stats")]
    if orientation_stats_list:
        orientation_html += "<h2>Dynamical Orientation Probe Statistics (Averaged across runs)</h2>"
        orientation_html += "<p><em>Delta > 0 indicates confidence was successfully destroyed. Larger delta is better.</em></p>"
        orientation_html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
        orientation_html += "<tr style='background-color: #f2f2f2;'><th>Category</th><th>Metric</th><th>Value (Mean)</th></tr>"
        
        # Aggregate
        keys_ps = orientation_stats_list[0]["per_sample"].keys()
        for k in keys_ps:
            avg_val = np.nanmean([s["per_sample"][k] for s in orientation_stats_list if not np.isnan(s["per_sample"][k])])
            val_str = f"{avg_val:.4f}" if not np.isnan(avg_val) else "N/A"
            orientation_html += f"<tr><td>Per-Sample</td><td>{k}</td><td>{val_str}</td></tr>"
            
        keys_gb = orientation_stats_list[0]["global"].keys()
        for k in keys_gb:
            avg_val = np.nanmean([s["global"][k] for s in orientation_stats_list if not np.isnan(s["global"][k])])
            val_str = f"{avg_val:.4f}" if not np.isnan(avg_val) else "N/A"
            orientation_html += f"<tr><td>Global</td><td>{k}</td><td>{val_str}</td></tr>"
        orientation_html += "</table><br><hr><br>"

    with open(args.out_html, 'w') as f:
        f.write("<html><head><title>AOPC Report</title></head><body style='font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;'>")
        f.write("<h1>AOPC Evaluation Report</h1>")
        f.write(orientation_html)
        f.write(fig_aopc.to_html(full_html=False, include_plotlyjs='cdn'))
        
        if num_runs > 1:
            f.write("<br><br><hr><br>")
            f.write(fig_box.to_html(full_html=False, include_plotlyjs='cdn'))
            
        f.write("<br><br><hr><br>")
        if total_examples_to_plot > 0:
            f.write(fig_logits.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("</body></html>")
        
    print(f"Interactive HTML report generated at: {args.out_html}")

if __name__ == "__main__":
    main()
