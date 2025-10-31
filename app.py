import streamlit as st
import pandas as pd
import io
from streamlit_option_menu import option_menu
from optimizer import optimize_rake_plan
from ml_rules import predict_demand_category

# Page configuration
st.set_page_config(
    page_title="UtkarshRake Optimizer",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<p class="main-header">🚂 UtkarshRake: Rake Formation Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered Decision Support System for SAIL Logistics</p>', unsafe_allow_html=True)

# Sidebar navigation with option menu
with st.sidebar:
    st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=SAIL+Logistics", use_container_width=True)
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Data Upload", "View Data", "Optimization", "Analytics"],
        icons=["house", "cloud-upload", "table", "gear", "graph-up"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#262730"},
            "icon": {"color": "#fafafa", "font-size": "18px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "margin": "0px",
                "color": "#fafafa",
                "--hover-color": "#3d3d4d",
            },
            "nav-link-selected": {"background-color": "#1f77b4", "color": "#ffffff"},
            "menu-title": {"color": "#fafafa"}
        }
    )

# Initialize session state for data persistence
if 'orders_df' not in st.session_state:
    st.session_state.orders_df = None
if 'wagons_df' not in st.session_state:
    st.session_state.wagons_df = None
if 'yard_df' not in st.session_state:
    st.session_state.yard_df = None
if 'optimization_complete' not in st.session_state:
    st.session_state.optimization_complete = False
if 'plan' not in st.session_state:
    st.session_state.plan = None

# HOME PAGE
if selected == "Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("### 📦 Order Management\nManage and track orders efficiently")
    with col2:
        st.success("### 🚃 Wagon Allocation\nOptimize wagon utilization")
    with col3:
        st.warning("### 🏭 Yard Capacity\nMonitor yard resources")
    
    st.markdown("---")
    st.subheader("🎯 Key Features")
    
    features_col1, features_col2 = st.columns(2)
    with features_col1:
        st.markdown("""
        - ✅ **AI-Powered Optimization**: Smart rake formation algorithms
        - ✅ **Real-time Analytics**: Monitor performance metrics
        - ✅ **Demand Prediction**: ML-based demand categorization
        """)
    with features_col2:
        st.markdown("""
        - ✅ **Capacity Management**: Track wagon and yard utilization
        - ✅ **Export Reports**: Download detailed optimization plans
        - ✅ **Interactive UI**: User-friendly interface
        """)
    
    st.markdown("---")
    st.info("👈 **Get Started**: Use the navigation menu to upload data and optimize your rake plan")

# DATA UPLOAD PAGE
elif selected == "Data Upload":
    st.header("📁 Upload Your Data Files")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📦 Orders")
        orders_file = st.file_uploader("Upload Orders CSV", type="csv", key="orders")
        if orders_file:
            st.success("✅ Orders file uploaded")
    
    with col2:
        st.subheader("🚃 Wagons")
        wagons_file = st.file_uploader("Upload Wagons CSV", type="csv", key="wagons")
        if wagons_file:
            st.success("✅ Wagons file uploaded")
    
    with col3:
        st.subheader("🏭 Yard")
        yard_file = st.file_uploader("Upload Yard CSV", type="csv", key="yard")
        if yard_file:
            st.success("✅ Yard file uploaded")
    
    st.markdown("---")
    
    if st.button("🔄 Load Data", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading data..."):
                # Load uploaded files or fallback to default
                st.session_state.orders_df = pd.read_csv(orders_file) if orders_file else pd.read_csv("order.csv")
                st.session_state.wagons_df = pd.read_csv(wagons_file) if wagons_file else pd.read_csv("wagons.csv")
                st.session_state.yard_df = pd.read_csv(yard_file) if yard_file else pd.read_csv("yard.csv")
                
                # Apply ML logic
                st.session_state.orders_df = predict_demand_category(st.session_state.orders_df)
                
                st.success("✅ Data loaded successfully! Go to 'View Data' to see your data.")
        except FileNotFoundError:
            st.error("⚠️ Default CSV files not found. Please upload all required files.")
        except Exception as e:
            st.error(f"🚨 Error loading data: {str(e)}")

# VIEW DATA PAGE
elif selected == "View Data":
    st.header("📊 Data Overview")
    
    if st.session_state.orders_df is None:
        st.warning("⚠️ No data loaded. Please upload files in the 'Data Upload' section.")
    else:
        tab1, tab2, tab3 = st.tabs(["📦 Orders", "🚃 Wagons", "🏭 Yard"])
        
        with tab1:
            st.subheader("Orders with Demand Category")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                if 'demand_category' in st.session_state.orders_df.columns:
                    demand_filter = st.multiselect(
                        "Filter by Demand Category",
                        options=st.session_state.orders_df['demand_category'].unique(),
                        default=st.session_state.orders_df['demand_category'].unique()
                    )
            with col2:
                if 'material' in st.session_state.orders_df.columns:
                    material_filter = st.multiselect(
                        "Filter by Material",
                        options=st.session_state.orders_df['material'].unique(),
                        default=st.session_state.orders_df['material'].unique()
                    )
            
            # Apply filters
            filtered_orders = st.session_state.orders_df.copy()
            if 'demand_category' in filtered_orders.columns:
                filtered_orders = filtered_orders[filtered_orders['demand_category'].isin(demand_filter)]
            if 'material' in filtered_orders.columns:
                filtered_orders = filtered_orders[filtered_orders['material'].isin(material_filter)]
            
            st.dataframe(filtered_orders, use_container_width=True, height=400)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Orders", len(filtered_orders))
            col2.metric("Total Weight", f"{filtered_orders['weight'].sum():.2f} tons")
            col3.metric("Avg Order Weight", f"{filtered_orders['weight'].mean():.2f} tons")
        
        with tab2:
            st.subheader("Available Wagons")
            st.dataframe(st.session_state.wagons_df, use_container_width=True, height=400)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Wagons", len(st.session_state.wagons_df))
            col2.metric("Total Capacity", f"{st.session_state.wagons_df['capacity'].sum():.2f} tons")
            col3.metric("Avg Capacity", f"{st.session_state.wagons_df['capacity'].mean():.2f} tons")
        
        with tab3:
            st.subheader("Yard Information")
            st.dataframe(st.session_state.yard_df, use_container_width=True, height=400)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Yards", len(st.session_state.yard_df))
            col2.metric("Total Rake Capacity", st.session_state.yard_df['max_rakes'].sum())
            col3.metric("Avg Rake Capacity", f"{st.session_state.yard_df['max_rakes'].mean():.2f}")

# OPTIMIZATION PAGE
elif selected == "Optimization":
    st.header("⚙️ Rake Formation Optimization")
    
    if st.session_state.orders_df is None:
        st.warning("⚠️ No data loaded. Please upload files in the 'Data Upload' section first.")
    else:
        st.info("📊 Ready to optimize! Review the summary below and click 'Run Optimization'.")
        
        # Summary before optimization
        col1, col2, col3 = st.columns(3)
        col1.metric("Orders to Process", len(st.session_state.orders_df))
        col2.metric("Available Wagons", len(st.session_state.wagons_df))
        col3.metric("Total Order Weight", f"{st.session_state.orders_df['weight'].sum():.2f} tons")
        
        st.markdown("---")
        
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("🔄 Optimizing rake formation..."):
                try:
                    # Convert to dictionaries
                    orders = dict(zip(st.session_state.orders_df['order_id'], 
                                    st.session_state.orders_df['weight']))
                    wagons = dict(zip(st.session_state.wagons_df['wagon_id'], 
                                    st.session_state.wagons_df['capacity']))
                    
                    # Run optimization
                    st.session_state.plan = optimize_rake_plan(orders, wagons)
                    st.session_state.optimization_complete = True
                    
                    if st.session_state.plan:
                        st.success("✨ Optimization Complete!")
                        st.balloons()
                    else:
                        st.error("❌ Could not find a feasible solution!")
                        
                except Exception as e:
                    st.error(f"🚨 Optimization error: {str(e)}")
        
        # Display results if optimization is complete
        if st.session_state.optimization_complete and st.session_state.plan:
            st.markdown("---")
            st.subheader("📊 Optimization Results")
            
            # Group by wagon
            wagon_assignments = {}
            for order_id, wagon_id in st.session_state.plan:
                wagon_assignments.setdefault(wagon_id, []).append(order_id)
            
            orders = dict(zip(st.session_state.orders_df['order_id'], 
                            st.session_state.orders_df['weight']))
            wagons = dict(zip(st.session_state.wagons_df['wagon_id'], 
                            st.session_state.wagons_df['capacity']))
            
            # Summary metrics
            total_utilization = 0
            overloaded_wagons = 0
            underutilized_wagons = 0
            
            for wagon_id, assigned_orders in wagon_assignments.items():
                total_weight = sum(orders[order_id] for order_id in assigned_orders)
                utilization = (total_weight / wagons[wagon_id]) * 100
                total_utilization += utilization
                
                if total_weight > wagons[wagon_id]:
                    overloaded_wagons += 1
                elif utilization < 50:
                    underutilized_wagons += 1
            
            avg_utilization = total_utilization / len(wagon_assignments) if wagon_assignments else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Wagons Used", len(wagon_assignments))
            col2.metric("Avg Utilization", f"{avg_utilization:.2f}%")
            col3.metric("⚠️ Overloaded", overloaded_wagons)
            col4.metric("⚠️ Underutilized", underutilized_wagons)
            
            st.markdown("---")
            
            # Create plan data for download
            plan_data = []
            
            # Display detailed assignments
            for wagon_id, assigned_orders in wagon_assignments.items():
                with st.expander(f"🚃 Wagon {wagon_id}", expanded=False):
                    total_weight = sum(orders[order_id] for order_id in assigned_orders)
                    remaining_capacity = wagons[wagon_id] - total_weight
                    utilization = round((total_weight / wagons[wagon_id]) * 100, 2)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Capacity", f"{wagons[wagon_id]} tons")
                    col2.metric("Used", f"{total_weight} tons")
                    col3.metric("Remaining", f"{remaining_capacity} tons")
                    col4.metric("Utilization", f"{utilization}%")

                    if remaining_capacity < 0:
                        st.error("⚠️ Warning: Wagon is overloaded!")
                    elif utilization < 50:
                        st.warning("⚠️ Warning: Wagon is underutilized")

                    st.write("📦 **Assigned Orders:**")
                    for order_id in assigned_orders:
                        order_info = st.session_state.orders_df[
                            st.session_state.orders_df['order_id'] == order_id
                        ].iloc[0]
                        st.write(f"- **{order_id}**: {order_info['material']} ({order_info['weight']} tons) | Demand: {order_info['demand_category']}")
                        
                        plan_data.append({
                            "Order ID": order_id,
                            "Wagon ID": wagon_id,
                            "Material": order_info['material'],
                            "Weight": order_info['weight'],
                            "Demand Category": order_info['demand_category'],
                            "Wagon Utilization": f"{utilization}%"
                        })
            
            # Download button
            if plan_data:
                st.markdown("---")
                output_df = pd.DataFrame(plan_data)
                csv = output_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Detailed Plan CSV",
                    data=csv,
                    file_name="rake_plan.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

# ANALYTICS PAGE
elif selected == "Analytics":
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.orders_df is None:
        st.warning("⚠️ No data loaded. Please upload files in the 'Data Upload' section.")
    else:
        tab1, tab2 = st.tabs(["📊 Data Analytics", "🎯 Optimization Insights"])
        
        with tab1:
            st.subheader("Order Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'demand_category' in st.session_state.orders_df.columns:
                    st.write("**Orders by Demand Category**")
                    demand_counts = st.session_state.orders_df['demand_category'].value_counts()
                    st.bar_chart(demand_counts)
            
            with col2:
                if 'material' in st.session_state.orders_df.columns:
                    st.write("**Orders by Material Type**")
                    material_counts = st.session_state.orders_df['material'].value_counts()
                    st.bar_chart(material_counts)
            
            st.markdown("---")
            st.subheader("Weight Distribution")
            st.line_chart(st.session_state.orders_df['weight'])
        
        with tab2:
            if not st.session_state.optimization_complete:
                st.info("⚠️ Run optimization first to see insights!")
            else:
                st.subheader("Optimization Performance")
                st.success("📊 Optimization insights will be displayed here after running optimization")
                
                # You can add more detailed analytics here based on optimization results

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚂 UtkarshRake - Powered by AI | © 2024 SAIL Logistics</p>
    </div>
    """,
    unsafe_allow_html=True
)