# -*- coding: utf-8 -*-
"""
Edit Trace Visualizer Module
Interactive visualization of edit traces and backend performance
"""
import numpy as np
from typing import Dict, List, Optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available. Install with: pip install plotly")


class EditTraceVisualizer:
    """Visualize edit traces and performance"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.figures = []
    
    def create_dashboard(
        self,
        edit_logs: List[Dict],
        backends: List[str] = None,
        languages: List[str] = None
    ):
        """
        Create interactive dashboard
        
        Args:
            edit_logs: List of edit logs
            backends: Backends to visualize
            languages: Languages to include
            
        Returns:
            Plotly figure or None
        """
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly required for visualization")
            return None
        
        backends = backends or ['russian', 'ibm']
        languages = languages or ['en', 'ru', 'es', 'fr', 'de']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Backend Performance Comparison',
                'Language-Specific Accuracy',
                'Edit Type Distribution',
                'Latency Distribution'
            )
        )
        
        # 1. Backend performance
        backend_data = self._aggregate_by_backend(edit_logs, backends)
        fig.add_trace(
            go.Bar(
                x=list(backend_data.keys()),
                y=[d['success_rate'] for d in backend_data.values()],
                name='Success Rate'
            ),
            row=1, col=1
        )
        
        # 2. Language accuracy
        lang_data = self._aggregate_by_language(edit_logs, languages)
        fig.add_trace(
            go.Bar(
                x=list(lang_data.keys()),
                y=[d['accuracy'] for d in lang_data.values()],
                name='Accuracy'
            ),
            row=1, col=2
        )
        
        # 3. Edit type distribution
        type_data = self._aggregate_by_type(edit_logs)
        fig.add_trace(
            go.Pie(
                labels=list(type_data.keys()),
                values=list(type_data.values()),
                name='Edit Types'
            ),
            row=2, col=1
        )
        
        # 4. Latency distribution
        latencies = [log.get('latency_ms', 0) for log in edit_logs]
        fig.add_trace(
            go.Histogram(
                x=latencies,
                name='Latency'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Quantum Backend Performance Dashboard",
            showlegend=True,
            height=800
        )
        
        self.figures.append(fig)
        return fig
    
    def create_heatmap(
        self,
        edit_logs: List[Dict],
        x_axis: str = 'language',
        y_axis: str = 'domain'
    ):
        """Create performance heatmap"""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Aggregate data
        matrix = self._create_matrix(edit_logs, x_axis, y_axis)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix['values'],
            x=matrix['x_labels'],
            y=matrix['y_labels'],
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=f'Performance Heatmap: {y_axis.title()} vs {x_axis.title()}',
            xaxis_title=x_axis.title(),
            yaxis_title=y_axis.title()
        )
        
        self.figures.append(fig)
        return fig
    
    def show_dashboard(self):
        """Display all created figures"""
        if not self.figures:
            print("⚠️  No figures to display. Create dashboard first.")
            return
        
        for fig in self.figures:
            if fig:
                fig.show()
    
    def _aggregate_by_backend(
        self,
        logs: List[Dict],
        backends: List[str]
    ) -> Dict:
        """Aggregate metrics by backend"""
        data = {}
        for backend in backends:
            backend_logs = [l for l in logs if l.get('backend') == backend]
            if backend_logs:
                success = sum(1 for l in backend_logs if l.get('success', False))
                data[backend] = {
                    'success_rate': success / len(backend_logs),
                    'count': len(backend_logs)
                }
        return data
    
    def _aggregate_by_language(
        self,
        logs: List[Dict],
        languages: List[str]
    ) -> Dict:
        """Aggregate metrics by language"""
        data = {}
        for lang in languages:
            lang_logs = [l for l in logs if l.get('lang') == lang]
            if lang_logs:
                success = sum(1 for l in lang_logs if l.get('success', False))
                data[lang] = {
                    'accuracy': success / len(lang_logs),
                    'count': len(lang_logs)
                }
        return data
    
    def _aggregate_by_type(self, logs: List[Dict]) -> Dict:
        """Aggregate by edit type"""
        types = {}
        for log in logs:
            edit_type = log.get('type', 'unknown')
            types[edit_type] = types.get(edit_type, 0) + 1
        return types
    
    def _create_matrix(
        self,
        logs: List[Dict],
        x_axis: str,
        y_axis: str
    ) -> Dict:
        """Create matrix for heatmap"""
        # Get unique values
        x_values = sorted(set(log.get(x_axis, 'unknown') for log in logs))
        y_values = sorted(set(log.get(y_axis, 'unknown') for log in logs))
        
        # Create matrix
        matrix = np.zeros((len(y_values), len(x_values)))
        
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                matching = [
                    l for l in logs
                    if l.get(x_axis) == x_val and l.get(y_axis) == y_val
                ]
                if matching:
                    success = sum(1 for l in matching if l.get('success', False))
                    matrix[i, j] = success / len(matching)
        
        return {
            'values': matrix.tolist(),
            'x_labels': x_values,
            'y_labels': y_values
        }


# Convenience function
def quick_visualize(edit_logs: List[Dict]):
    """Quick visualization"""
    viz = EditTraceVisualizer()
    dashboard = viz.create_dashboard(edit_logs)
    if dashboard:
        dashboard.show()
