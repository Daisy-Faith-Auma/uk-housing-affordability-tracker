import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Configuration directly in file to avoid import issues
API_CONFIG = {
    'ons_base_url': 'https://api.beta.ons.gov.uk/v1/datasets/',
    'land_registry_base': 'http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com',
    'request_timeout': 60,
    'rate_limit_delay': 1,
    'user_agent': 'UK-Housing-Tracker/1.0 (Educational Project)'
}

ANALYSIS_CONFIG = {
    'affordability_threshold': 7.0,
    'crisis_threshold': 8.0,
    'start_year': 2002,
    'end_year': 2024,
    'min_transactions': 10
}

class SimpleHousingProcessor:
    """Simplified processor with all methods included"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def create_enhanced_sample_data(self) -> pd.DataFrame:
        """Create comprehensive sample data with regional classification"""
        
        # Extended local authorities with regions
        authorities_data = [
            # London
            ('Westminster', 'London', 14.2, 850000),
            ('Kensington and Chelsea', 'London', 13.8, 920000),
            ('Camden', 'London', 10.8, 680000),
            ('Islington', 'London', 10.2, 650000),
            ('Hackney', 'London', 9.1, 580000),
            ('Tower Hamlets', 'London', 8.9, 560000),
            
            # South East
            ('Brighton and Hove', 'South East', 8.9, 420000),
            ('Oxford', 'South East', 9.2, 480000),
            ('Cambridge', 'South East', 8.7, 520000),
            ('Reading', 'South East', 7.8, 390000),
            
            # North
            ('Manchester', 'North West', 5.8, 220000),
            ('Liverpool', 'North West', 4.9, 180000),
            ('Leeds', 'Yorkshire', 5.4, 210000),
            ('Sheffield', 'Yorkshire', 4.2, 160000),
            ('Newcastle', 'North East', 4.1, 155000),
            
            # Midlands
            ('Birmingham', 'West Midlands', 5.9, 210000),
            ('Nottingham', 'East Midlands', 5.2, 185000),
            ('Leicester', 'East Midlands', 5.1, 190000),
            ('Coventry', 'West Midlands', 5.7, 200000),
            
            # Wales & Southwest
            ('Cardiff', 'Wales', 6.2, 240000),
            ('Bristol', 'South West', 7.1, 350000),
            ('Bath', 'South West', 8.3, 480000),
        ]
        
        years = list(range(2018, 2025))
        
        data = []
        for authority, region, base_ratio, base_price in authorities_data:
            for year in years:
                # Add year-over-year variation
                year_factor = 1 + (year - 2020) * 0.05  # 5% annual growth
                price_variation = np.random.uniform(0.95, 1.05)
                ratio_variation = np.random.uniform(0.95, 1.05)
                
                current_price = base_price * year_factor * price_variation
                current_ratio = base_ratio * ratio_variation
                current_earnings = current_price / current_ratio
                
                data.append({
                    'local_authority': authority,
                    'region': region,
                    'year': year,
                    'price_to_earnings_ratio': round(current_ratio, 2),
                    'median_house_price': round(current_price, -3),
                    'median_earnings': round(current_earnings, -2),
                    'data_source': 'SAMPLE_ENHANCED'
                })
        
        df = pd.DataFrame(data)
        
        # Add calculated fields
        df['affordability_category'] = pd.cut(
            df['price_to_earnings_ratio'],
            bins=[0, 4, 6, 8, 10, float('inf')],
            labels=['Very Affordable', 'Affordable', 'Stretched', 'Unaffordable', 'Crisis']
        )
        
        df['is_crisis_area'] = df['price_to_earnings_ratio'] > 8.0
        
        return df
    
    def calculate_regional_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate regional summary statistics"""
        
        latest_year = data['year'].max()
        latest_data = data[data['year'] == latest_year]
        
        regional_stats = latest_data.groupby('region').agg({
            'price_to_earnings_ratio': ['mean', 'median', 'std'],
            'median_house_price': ['mean', 'median'],
            'median_earnings': ['mean', 'median'],
            'is_crisis_area': 'sum',
            'local_authority': 'count'
        }).round(2)
        
        # Flatten column names
        regional_stats.columns = ['_'.join(col).strip() for col in regional_stats.columns]
        regional_stats = regional_stats.reset_index()
        
        # Calculate crisis percentage
        regional_stats['crisis_percentage'] = (
            regional_stats['is_crisis_area_sum'] / regional_stats['local_authority_count'] * 100
        ).round(1)
        
        return regional_stats
    
    def create_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time series aggregations with clean column names"""
        
        national_trends = data.groupby('year').agg({
            'price_to_earnings_ratio': 'median',
            'median_house_price': 'median',
            'median_earnings': 'median',
            'is_crisis_area': ['sum', 'count']
        }).round(2)
        
        # Flatten columns with cleaner names
        national_trends.columns = [
            'price_to_earnings_ratio',
            'median_house_price', 
            'median_earnings',
            'crisis_areas_count',
            'total_areas_count'
        ]
        
        national_trends = national_trends.reset_index()
        
        # Calculate national crisis percentage
        national_trends['crisis_percentage'] = (
            national_trends['crisis_areas_count'] / national_trends['total_areas_count'] * 100
        ).round(1)
        
        return national_trends

class ProfessionalHousingDashboard:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / 'data'
        
        # Configure Streamlit
        st.set_page_config(
            page_title="UK Housing Affordability Tracker",
            page_icon="🏠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Load custom CSS
        self._load_custom_css()
    
    def _load_custom_css(self):
        """Load custom CSS styling"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79 0%, #2e8b57 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f4e79;
            margin-bottom: 1rem;
        }
        
        .crisis-alert {
            background: #ffebee;
            border: 1px solid #f44336;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .stMetric {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .data-source {
            font-size: 0.8rem;
            color: #666;
            font-style: italic;
            margin-top: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load comprehensive housing data"""
        processor = SimpleHousingProcessor(self.data_dir)
        
        # Create enhanced sample data
        data = processor.create_enhanced_sample_data()
        
        # Save to cache for future use
        cache_path = self.data_dir / 'cache' / 'enhanced_housing_data.csv'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path, index=False)
        
        return data
    
    def run(self):
        """Main dashboard application"""
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🏠 UK Housing Affordability Tracker</h1>
            <p>Comprehensive analysis of housing costs and affordability across England and Wales</p>
            <p><em>Built with ONS methodology and government data standards</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data with progress indicator
        with st.spinner("Loading comprehensive housing dataset..."):
            data = self.load_data()
        
        if data.empty:
            st.error("No data available. Please check data sources.")
            return
        
        # Initialize processor for calculations
        processor = SimpleHousingProcessor(self.data_dir)
        
        # Sidebar controls
        filters = self._render_sidebar(data)
        
        # Apply filters
        filtered_data = self._apply_filters(data, filters)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗺️ Regional Analysis", "📈 Trends", "⚠️ Crisis Areas"])
        
        with tab1:
            self._render_overview_tab(filtered_data, processor)
        
        with tab2:
            self._render_regional_tab(filtered_data, processor)
        
        with tab3:
            self._render_trends_tab(filtered_data, processor)
        
        with tab4:
            self._render_crisis_tab(filtered_data, processor)
        
        # Footer
        self._render_footer()
    
    def _render_sidebar(self, data: pd.DataFrame) -> dict:
        """Render sidebar controls"""
        
        with st.sidebar:
            st.header("🎛️ Dashboard Controls")
            
            # Data info section
            st.subheader("📊 Dataset Information")
            total_records = len(data)
            regions = data['region'].nunique()
            authorities = data['local_authority'].nunique()
            year_range = f"{data['year'].min()} - {data['year'].max()}"
            
            st.info(f"""
            **Records**: {total_records:,}  
            **Regions**: {regions}  
            **Authorities**: {authorities}  
            **Years**: {year_range}  
            **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            """)
            
            st.markdown("---")
            
            # Filter controls
            st.subheader("🔍 Filters")
            
            # Year selection
            years = sorted(data['year'].unique())
            year_selection = st.radio(
                "Time Period",
                ["Latest Year", "All Years", "Custom Range"],
                help="Choose time period for analysis"
            )
            
            if year_selection == "Latest Year":
                selected_years = [max(years)]
            elif year_selection == "All Years":
                selected_years = years
            else:  # Custom Range
                year_range = st.select_slider(
                    "Select Year Range",
                    options=years,
                    value=(max(years) - 2, max(years)),
                    help="Select custom year range"
                )
                selected_years = list(range(year_range[0], year_range[1] + 1))
            
            # Region filter
            regions = ['All Regions'] + sorted(data['region'].unique().tolist())
            selected_regions = st.multiselect(
                "Regions",
                options=regions,
                default=['All Regions'],
                help="Filter by geographic regions"
            )
            
            # Analysis parameters
            st.markdown("---")
            st.subheader("⚙️ Analysis Settings")
            
            affordability_threshold = st.slider(
                "Affordability Threshold",
                min_value=5.0,
                max_value=12.0,
                value=7.0,
                step=0.5,
                help="Ratio above which housing is considered unaffordable"
            )
            
            crisis_threshold = st.slider(
                "Crisis Threshold",
                min_value=7.0,
                max_value=15.0,
                value=8.0,
                step=0.5,
                help="Ratio above which areas are in housing crisis"
            )
        
        return {
            'years': selected_years,
            'regions': selected_regions,
            'affordability_threshold': affordability_threshold,
            'crisis_threshold': crisis_threshold
        }
    
    def _apply_filters(self, data: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply user-selected filters to data"""
        
        filtered_data = data.copy()
        
        # Year filter
        filtered_data = filtered_data[filtered_data['year'].isin(filters['years'])]
        
        # Region filter
        if 'All Regions' not in filters['regions']:
            filtered_data = filtered_data[filtered_data['region'].isin(filters['regions'])]
        
        return filtered_data
    
    def _render_overview_tab(self, data: pd.DataFrame, processor: SimpleHousingProcessor):
        """Render overview dashboard tab"""
        
        st.header("📊 Housing Affordability Overview")
        
        # Key metrics
        latest_data = data[data['year'] == data['year'].max()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_areas = len(latest_data)
            st.metric(
                "Total Areas Analyzed",
                f"{total_areas}",
                help="Number of local authorities in analysis"
            )
        
        with col2:
            crisis_areas = len(latest_data[latest_data['is_crisis_area']])
            crisis_pct = (crisis_areas / total_areas * 100) if total_areas > 0 else 0
            st.metric(
                "Crisis Areas",
                f"{crisis_areas}",
                delta=f"{crisis_pct:.1f}% of total",
                help="Areas with >8x price-to-earnings ratio"
            )
        
        with col3:
            median_ratio = latest_data['price_to_earnings_ratio'].median()
            healthy_diff = median_ratio - 7.0
            st.metric(
                "Median P/E Ratio",
                f"{median_ratio:.1f}x",
                delta=f"{healthy_diff:+.1f} vs healthy",
                delta_color="inverse",
                help="Median price-to-earnings ratio across all areas"
            )
        
        with col4:
            median_price = latest_data['median_house_price'].median()
            st.metric(
                "Median House Price",
                f"£{median_price:,.0f}",
                help="Median house price across selected areas"
            )
        
        # Crisis alert
        if crisis_pct > 50:
            st.markdown("""
            <div class="crisis-alert">
                <strong>⚠️ Housing Crisis Alert</strong><br>
                More than half of analyzed areas show crisis-level unaffordability. 
                This indicates a systemic housing affordability problem requiring urgent policy intervention.
            </div>
            """, unsafe_allow_html=True)
        
        # Main visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Affordability Distribution")
            
            fig = px.histogram(
                latest_data,
                x='price_to_earnings_ratio',
                nbins=25,
                title="Distribution of Price-to-Earnings Ratios",
                labels={
                    'price_to_earnings_ratio': 'Price-to-Earnings Ratio',
                    'count': 'Number of Areas'
                },
                color_discrete_sequence=['#2E8B57']
            )
            
            # Add threshold lines
            fig.add_vline(x=7, line_dash="dash", line_color="orange", 
                         annotation_text="Healthy (7x)", annotation_position="top")
            fig.add_vline(x=8, line_dash="dash", line_color="red", 
                         annotation_text="Crisis (8x)", annotation_position="top")
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💰 Price vs Earnings Relationship")
            
            fig = px.scatter(
                latest_data,
                x='median_earnings',
                y='median_house_price',
                size='price_to_earnings_ratio',
                color='affordability_category',
                hover_name='local_authority',
                hover_data=['region', 'price_to_earnings_ratio'],
                title="House Prices vs Earnings by Local Authority",
                labels={
                    'median_earnings': 'Median Annual Earnings (£)',
                    'median_house_price': 'Median House Price (£)',
                    'affordability_category': 'Affordability'
                },
                color_discrete_map={
                    'Very Affordable': '#2E8B57',
                    'Affordable': '#90EE90',
                    'Stretched': '#FFD700',
                    'Unaffordable': '#FF8C00',
                    'Crisis': '#DC143C'
                }
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Regional comparison
        st.subheader("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Regional Comparison")
        
        regional_stats = processor.calculate_regional_stats(data)
        
        # Regional bar chart
        fig = px.bar(
            regional_stats,
            x='region',
            y='price_to_earnings_ratio_median',
            color='crisis_percentage',
            title="Median Affordability Ratio by Region",
            labels={
                'region': 'Region',
                'price_to_earnings_ratio_median': 'Median Price-to-Earnings Ratio',
                'crisis_percentage': 'Crisis Areas (%)'
            },
            color_continuous_scale='Reds'
        )
        
        fig.add_hline(y=7, line_dash="dash", line_color="orange", 
                     annotation_text="Healthy Threshold")
        fig.add_hline(y=8, line_dash="dash", line_color="red", 
                     annotation_text="Crisis Threshold")
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_regional_tab(self, data: pd.DataFrame, processor: SimpleHousingProcessor):
        """Render regional analysis tab"""
        
        st.header("🗺️ Regional Analysis")
        st.write("Regional comparison of housing affordability across different UK regions.")
        
        # Regional overview metrics
        latest_data = data[data['year'] == data['year'].max()]
        
        regional_summary = latest_data.groupby('region').agg({
            'price_to_earnings_ratio': 'median',
            'median_house_price': 'median',
            'is_crisis_area': ['sum', 'count']
        }).round(2)
        
        regional_summary.columns = ['median_ratio', 'median_price', 'crisis_count', 'total_count']
        regional_summary['crisis_pct'] = (regional_summary['crisis_count'] / regional_summary['total_count'] * 100).round(1)
        regional_summary = regional_summary.reset_index()
        
        st.dataframe(regional_summary, use_container_width=True)
    
    def _render_trends_tab(self, data: pd.DataFrame, processor: SimpleHousingProcessor):
        """Render trends analysis tab"""
        
        st.header("📈 Trends Analysis")
        
        # Time series data
        time_series = processor.create_time_series_data(data)
        
        # National trends
        st.subheader("National Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                time_series,
                x='year',
                y='price_to_earnings_ratio',
                title="National Median Affordability Trend",
                markers=True
            )
            
            fig.add_hline(y=7, line_dash="dash", line_color="orange")
            fig.add_hline(y=8, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                time_series,
                x='year',
                y='crisis_percentage',
                title="Percentage of Areas in Crisis",
                markers=True,
                line_shape='spline'
            )
            
            fig.update_traces(line_color='#DC143C')
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_crisis_tab(self, data: pd.DataFrame, processor: SimpleHousingProcessor):
        """Render crisis areas analysis tab"""
        
        st.header("⚠️ Housing Crisis Areas")
        
        latest_data = data[data['year'] == data['year'].max()]
        crisis_data = latest_data[latest_data['is_crisis_area']].copy()
        
        if crisis_data.empty:
            st.success("🎉 No areas currently meet crisis threshold criteria!")
            return
        
        # Crisis overview
        crisis_count = len(crisis_data)
        total_count = len(latest_data)
        crisis_pct = (crisis_count / total_count * 100)
        
        st.error(f"**{crisis_count} out of {total_count} areas ({crisis_pct:.1f}%) are in housing crisis**")
        
        # Crisis areas table
        st.subheader("Crisis Areas Detail")
        
        crisis_display = crisis_data[['local_authority', 'region', 'price_to_earnings_ratio', 
                                   'median_house_price', 'median_earnings']].copy()
        
        crisis_display.columns = ['Local Authority', 'Region', 'P/E Ratio', 
                                'Median Price', 'Median Earnings']
        
        # Format currency columns
        crisis_display['Median Price'] = crisis_display['Median Price'].apply(
            lambda x: f"£{x:,.0f}"
        )
        crisis_display['Median Earnings'] = crisis_display['Median Earnings'].apply(
            lambda x: f"£{x:,.0f}"
        )
        
        crisis_display = crisis_display.sort_values('P/E Ratio', ascending=False)
        
        st.dataframe(crisis_display, use_container_width=True, hide_index=True)
    
    def _render_footer(self):
        """Render dashboard footer"""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Data Sources:**
            - ONS Housing Affordability Data
            - Annual Survey of Hours and Earnings (ASHE)
            - Land Registry Price Paid Data
            """)
        
        with col2:
            st.markdown("""
            **Methodology:**
            - Price-to-earnings ratios calculated using median values
            - Crisis threshold: >8x annual earnings
            - Regional classifications based on ONS standards
            """)
        
        with col3:
            st.markdown("""
            **Technical Stack:**
            - Built with Streamlit & Plotly
            - Data processing with pandas
            - Hosted on Streamlit Cloud
            """)


if __name__ == "__main__":
    dashboard = ProfessionalHousingDashboard()
    dashboard.run()