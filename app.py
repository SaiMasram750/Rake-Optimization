import streamlit as st
import pandas as pd
import io
from optimizer import optimize_rake_plan
from ml_rules import predict_demand_category  # Optional ML logic
st.title("UtkarshRake : Rake Formation Optimiser")
st.markdown("#### AI-powered Decision Support System for SAIL Logistics")


st.sidebar.header("📁 Upload Your CSV Files")
orders_file = st.sidebar.file_uploader("Upload Orders CSV", type="csv")
wagons_file = st.sidebar.file_uploader("Upload Wagons CSV", type="csv")
yard_file = st.sidebar.file_uploader("Upload Yard CSV", type="csv")

try:
    # Load uploaded files or fallback to default
    orders_df = pd.read_csv(orders_file) if orders_file else pd.read_csv("order.csv")
    wagons_df = pd.read_csv(wagons_file) if wagons_file else pd.read_csv("wagons.csv")
    yard_df = pd.read_csv(yard_file) if yard_file else pd.read_csv("yard.csv")

    # Apply rule-based ML logic
    orders_df = predict_demand_category(orders_df)

    # Display raw data
    st.subheader("📦 Orders with Demand Category")
    st.dataframe(orders_df)

    st.subheader("🚃 Wagons")
    st.dataframe(wagons_df)

    st.subheader("🏭 Yard")
    st.dataframe(yard_df)

    # Convert to dictionaries for optimization
    orders = dict(zip(orders_df['order_id'], orders_df['weight']))
    wagons = dict(zip(wagons_df['wagon_id'], wagons_df['capacity']))
    yard_capacity = dict(zip(yard_df['yard_id'], yard_df['max_rakes']))

    if st.button("Optimize Rake Plan"):
        plan = optimize_rake_plan(orders, wagons)

        if plan:
            st.success("✨ Optimization Complete!")
            st.subheader("📊 Optimization Results")

            # Group by wagon
            wagon_assignments = {}
            for order_id, wagon_id in plan:
                wagon_assignments.setdefault(wagon_id, []).append(order_id)

            # Create DataFrame for download
            plan_data = []
            
            # Display assignments with metrics
            for wagon_id, assigned_orders in wagon_assignments.items():
                with st.expander(f"🚃 Wagon {wagon_id}"):
                    total_weight = sum(orders[order_id] for order_id in assigned_orders)
                    remaining_capacity = wagons[wagon_id] - total_weight
                    utilization = round((total_weight / wagons[wagon_id]) * 100, 2)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Capacity", f"{wagons[wagon_id]} tons")
                    col2.metric("Used", f"{total_weight} tons")
                    col3.metric("Remaining", f"{remaining_capacity} tons")
                    col4.metric("Utilization", f"{utilization}%")

                    # Check for overload or underutilization
                    if remaining_capacity < 0:
                        st.error("⚠️ Warning: Wagon is overloaded!")
                    elif utilization < 50:
                        st.warning("⚠️ Warning: Wagon is underutilized")

                    st.write("📦 Assigned Orders:")
                    for order_id in assigned_orders:
                        order_info = orders_df[orders_df['order_id'] == order_id].iloc[0]
                        st.write(f"- {order_id}: {order_info['material']} ({order_info['weight']} tons) | Demand: {order_info['demand_category']}")
                        
                        # Add to plan data for download
                        plan_data.append({
                            "Order ID": order_id,
                            "Wagon ID": wagon_id,
                            "Material": order_info['material'],
                            "Weight": order_info['weight'],
                            "Demand Category": order_info['demand_category'],
                            "Wagon Utilization": f"{utilization}%"
                        })
            
            # Create download button for the plan
            if plan_data:
                output_df = pd.DataFrame(plan_data)
                csv = output_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Detailed Plan CSV",
                    data=csv,
                    file_name="rake_plan.csv",
                    mime="text/csv"
                )
        else:
            st.error("❌ Could not find a feasible solution!")

except FileNotFoundError:
    st.error("⚠️ Missing CSV files. Please upload or ensure default files are present.")
except Exception as e:
    st.error(f"🚨 Unexpected error: {str(e)}")
