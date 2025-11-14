[TrendStock.README.md](https://github.com/user-attachments/files/23539098/TrendStock.README.md)
# TrendStock AI

**Sell smarter.** Product Intelligence Platform for Small Businesses

![TrendStock AI Logo](https://via.placeholder.com/800x200/0BA14A/FFFFFF?text=TrendStock+AI)

## 🎯 Project Overview

**Student:** Jostin Wilson (DF-R302-04)  
**Course:** AI 560-01 Applied AI Design and Development Lab  
**Institution:** Savannah College of Art and Design  
**Professor:** Danyl Bartlett  
**Version:** 1.0.0

TrendStock AI is a production-ready product intelligence platform that helps small businesses understand consumer search trends and optimize advertising spend. By analyzing Google Trends data, the platform provides actionable insights on seasonal product recommendations, budget allocation, and competitive positioning.

### Key Features

- 📊 **Real-Time Dashboard Analytics** - Interactive visualizations of product trends
- 🤖 **AI-Powered Seasonal Suggestions** - Quarterly product recommendations
- 💬 **Conversational Chatbot** - Natural language business intelligence
- 📈 **Product Analytics** - Detailed forecasting and trend analysis
- 🔍 **Competitor Intelligence** - Market positioning insights
- 🔄 **MLflow Integration** - Experiment tracking and model management

---

## 🏗️ Architecture

### System Components

```
TrendStock AI
├── Configuration Layer
│   └── TrendStockConfig
├── Data Generation Layer
│   ├── MockTrendsDataGenerator
│   └── TrendscopeClient
├── Analysis Engine
│   └── TrendAnalysisEngine
├── User Interface
│   ├── Gradio Web Interface
│   └── Interactive Visualizations
└── AI Assistant
    └── TrendStockChatbot
```

### Core Technologies

- **Frontend:** Gradio (Interactive Web Interface)
- **Visualization:** Plotly (Charts & Graphs)
- **Data Processing:** Pandas, NumPy
- **ML Tracking:** MLflow
- **AI Models:** HuggingFace (TrendsScope Analyzer Pro)
- **Development:** Python 3.8+, Jupyter Notebooks

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) GPU support for advanced features

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/trendstock-ai.git
cd trendstock-ai

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Requirements

```txt
gradio>=4.0.0
plotly>=5.0.0
pandas>=2.0.0
numpy>=1.24.0
mlflow>=2.0.0
requests>=2.31.0
```

---

## 🚀 Usage

### Running the Complete Application

```python
# Launch TrendStock AI
python app.py
```

The application will start on `http://localhost:7860`

### Using Jupyter Notebooks

For development and testing:

```bash
# Start Jupyter
jupyter notebook

# Open TREND_MAIN.ipynb
```

### Quick Demo (Notebook)

```python
import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Initialize configuration
class Config:
    def __init__(self):
        self.quarters = {
            'Q1': {'months': [1,2,3], 'name': 'Winter/Early Spring', 'emoji': '❄️'},
            'Q2': {'months': [4,5,6], 'name': 'Spring/Early Summer', 'emoji': '🌸'},
            'Q3': {'months': [7,8,9], 'name': 'Summer/Early Fall', 'emoji': '☀️'},
            'Q4': {'months': [10,11,12], 'name': 'Fall/Holiday Season', 'emoji': '🍂'}
        }
        self.categories = ['Apparel', 'Accessories', 'Footwear', 'Outerwear']
    
    def get_current_quarter(self):
        month = datetime.now().month
        for q, info in self.quarters.items():
            if month in info['months']:
                return q, info['name'], info['emoji']
        return 'Q1', 'Winter/Early Spring', '❄️'

config = Config()
print(f"Current Quarter: {config.get_current_quarter()}")
```

---

## 📊 Core Components

### 1. Configuration System

```python
class TrendStockConfig:
    """Central configuration for the application"""
    
    def __init__(self):
        self.app_name = "TrendStock AI"
        self.version = "1.0.0"
        
        # Quarterly seasonal definitions
        self.quarters = {
            'Q1': {'months': [1,2,3], 'name': 'Winter/Early Spring'},
            'Q2': {'months': [4,5,6], 'name': 'Spring/Early Summer'},
            'Q3': {'months': [7,8,9], 'name': 'Summer/Early Fall'},
            'Q4': {'months': [10,11,12], 'name': 'Fall/Holiday Season'}
        }
        
        # Product categories for retail
        self.categories = [
            'Apparel', 'Accessories', 'Footwear', 
            'Outerwear', 'Activewear', 'Formal Wear'
        ]
```

### 2. Data Generation

```python
class MockTrendsDataGenerator:
    """Generate realistic trend data with seasonal intelligence"""
    
    def __init__(self, config):
        self.config = config
        np.random.seed(42)
    
    def generate_products(self, quarter, num_products=50):
        """Generate product data with seasonal boosts"""
        
        # Seasonal boost factors
        boosts = {
            'Q1': {'Outerwear': 0.4, 'Formal Wear': 0.2},
            'Q2': {'Activewear': 0.5, 'Casual Wear': 0.3},
            'Q3': {'Activewear': 0.4, 'Footwear': 0.3},
            'Q4': {'Formal Wear': 0.6, 'Accessories': 0.5}
        }
        
        products = []
        for i in range(num_products):
            category = np.random.choice(self.config.categories)
            base_trend = np.random.uniform(25, 85)
            
            # Apply seasonal adjustments
            boost = boosts.get(quarter, {}).get(category, 0)
            trend_score = min(100, max(0, base_trend + (base_trend * boost)))
            
            products.append({
                'name': f"{category} Item {i:03d}",
                'category': category,
                'trend_score': round(trend_score, 1),
                'search_volume': int(trend_score * 1000),
                'forecast': self._generate_forecast(trend_score),
                'recommended_budget': round(40 + (trend_score * 8), 2)
            })
        
        return pd.DataFrame(products)
```

### 3. Analysis Engine

```python
class TrendAnalysisEngine:
    """Core analytics for product trends"""
    
    def __init__(self, data):
        self.data = data
    
    def get_opportunities(self, threshold=60):
        """Identify high-opportunity products"""
        opps = self.data[self.data['trend_score'] >= threshold].copy()
        
        # Calculate composite opportunity score
        opps['opportunity_score'] = (
            opps['trend_score'] * 0.4 + 
            opps['forecast'].apply(np.mean) * 0.6
        )
        
        return opps.sort_values('opportunity_score', ascending=False)
    
    def get_category_stats(self):
        """Aggregate statistics by category"""
        stats = self.data.groupby('category').agg({
            'trend_score': ['mean', 'std', 'count'],
            'search_volume': 'sum',
            'recommended_budget': 'sum'
        }).round(2)
        
        return stats
```

### 4. AI Chatbot

```python
class TrendStockChatbot:
    """Intelligent conversational assistant"""
    
    def __init__(self, config, engine, data):
        self.config = config
        self.engine = engine
        self.data = data
    
    def respond(self, message):
        """Process user query and generate response"""
        msg = message.lower()
        
        # Intent detection
        if any(word in msg for word in ['quarter', 'season', 'when']):
            return self._seasonal_response()
        
        elif any(word in msg for word in ['trend', 'hot', 'opportunity']):
            return self._trending_response()
        
        elif any(word in msg for word in ['budget', 'spend']):
            return self._budget_response()
        
        else:
            return self._help_response()
    
    def _trending_response(self):
        """Generate response about trending products"""
        opps = self.engine.get_opportunities().head(5)
        
        response = "🚀 **Top 5 High-Opportunity Products:**\n\n"
        for i, (_, row) in enumerate(opps.iterrows(), 1):
            response += f"**{i}. {row['name']}**\n"
            response += f"   Trend: {row['trend_score']:.1f}% | "
            response += f"Budget: ${row['recommended_budget']:.2f}\n\n"
        
        return response
```

### 5. Visualization Components

```python
def create_trend_scatter(data, category="All"):
    """Create interactive scatter plot of trends"""
    df = data.copy() if category == "All" else data[data['category'] == category]
    
    fig = px.scatter(
        df, 
        x='trend_score', 
        y='search_volume',
        size='recommended_budget',
        color='category',
        hover_name='name',
        title='Product Trend Analysis',
        labels={
            'trend_score': 'Trend Score (%)',
            'search_volume': 'Search Volume'
        }
    )
    
    fig.update_layout(height=500)
    return fig

def create_forecast_chart(data):
    """Generate 3-month forecast visualization"""
    top_products = data.nlargest(5, 'trend_score')
    
    fig = go.Figure()
    for _, row in top_products.iterrows():
        values = [row['trend_score']] + row['forecast']
        fig.add_trace(go.Scatter(
            x=['Now', 'Month 1', 'Month 2', 'Month 3'],
            y=values,
            mode='lines+markers',
            name=row['name'][:20],
            line=dict(width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title='3-Month Trend Forecast',
        xaxis_title='Period',
        yaxis_title='Trend Score (%)',
        height=400
    )
    
    return fig
```

---

## 🎨 User Interface

### Gradio Interface Structure

The application features a tabbed interface with four main sections:

1. **Dashboard** - Real-time analytics and visualizations
2. **AI Suggestions** - Seasonal product recommendations
3. **AI Assistant** - Conversational chatbot interface
4. **Product Intelligence** - Detailed product analytics
5. **Export & Tracking** - MLflow integration and reporting

### Brand Colors

```css
Primary Green: #0BA14A
Navy Blue: #141140
White: #FFFFFF
Background: #F5F5F5
```

---

## 🔬 MLflow Integration

### Experiment Tracking

```python
import mlflow

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("TrendStock_AI")

# Log parameters and metrics
with mlflow.start_run():
    mlflow.log_param("model_version", "1.0.0")
    mlflow.log_param("quarter", "Q4")
    mlflow.log_metric("avg_trend_score", 65.3)
    mlflow.log_metric("total_products", 50)
```

### Model Registration

The project includes model registration capabilities for production deployment through HP AI Studio.

---

## 📱 API Integration

### TrendsScope API Client

```python
class TrendscopeClient:
    """Client for Google Trends data via TrendsScope Analyzer Pro"""
    
    def __init__(self, api_url):
        self.api_url = api_url
        self.cache = {}
    
    def get_trend_data(self, keyword, timeframe='today 3-m', geo='US'):
        """Fetch real-time trend data"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "data": [keyword, timeframe, geo]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_response(data)
            else:
                return self._generate_fallback_data(keyword)
                
        except Exception as e:
            print(f"API Error: {e}")
            return self._generate_fallback_data(keyword)
```

---

## 📂 Project Structure

```
trendstock-ai/
│
├── app.py                          # Main application file
├── TREND_MAIN.ipynb               # Primary Jupyter notebook
├── TrendStock_AI_Complete.py      # Standalone Python version
├── THE_AI_TEMPLATE.ipynb          # HP AI Studio template
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── docs/                          # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── user_guide.md
│
├── data/                          # Data storage
│   ├── mock/                      # Mock data generators
│   └── cache/                     # API response cache
│
├── models/                        # MLflow models
│   └── registered/
│
└── tests/                         # Unit tests
    ├── test_config.py
    ├── test_engine.py
    └── test_chatbot.py
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_engine.py

# With coverage
pytest --cov=app tests/
```

### Example Test

```python
import pytest
from app import TrendAnalysisEngine

def test_get_opportunities():
    """Test opportunity detection"""
    data = pd.DataFrame({
        'trend_score': [70, 50, 80, 30],
        'forecast': [[75, 78, 80], [45, 43, 40], [85, 88, 90], [25, 22, 20]]
    })
    
    engine = TrendAnalysisEngine(data)
    opps = engine.get_opportunities(threshold=60)
    
    assert len(opps) == 2
    assert opps.iloc[0]['trend_score'] == 80
```

---

## 🚧 Development Roadmap

### Completed Features ✅
- [x] Core trend analysis engine
- [x] Seasonal intelligence system
- [x] Interactive Gradio interface
- [x] AI chatbot assistant
- [x] MLflow integration
- [x] Comprehensive visualizations
- [x] Mock data generation

### Planned Features 🔜
- [ ] Real-time Google Trends API integration
- [ ] User authentication system
- [ ] Database persistence (PostgreSQL)
- [ ] Advanced forecasting models (ARIMA, Prophet)
- [ ] Email report automation
- [ ] Mobile-responsive design
- [ ] Multi-language support
- [ ] Integration with e-commerce platforms

---

## 🤝 Contributing

We welcome contributions to TrendStock AI! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Write unit tests for new features

---

## 📄 License

This project is part of academic coursework at Savannah College of Art and Design (SCAD).

**Copyright © 2025 Jostin Wilson**

For academic use only. Not for commercial distribution.

---

## 🙏 Acknowledgments

- **Professor Danyl Bartlett** - Course instruction and guidance
- **HP AI Studio** - Development environment
- **HuggingFace** - TrendsScope Analyzer Pro model
- **SCAD AI 560-01** - Applied AI Design and Development Lab
- **Google Trends** - Data source inspiration

---

## 📞 Contact

**Jostin Wilson**  
Student ID: DF-R302-04  
Course: AI 560-01  
Savannah College of Art and Design

For questions or feedback about this project:
- 📧 Email: joztin.pro@gmail.com
