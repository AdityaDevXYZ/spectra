import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import umap
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pinn_model import TabularAttentionNetwork, physics_informed_loss

def main():
    print("🚀 Initializing Systems-Level Ingestion Engine...")
    
    try:
        # Import our compiled Rust module
        import rust_ingest
        df_polars = rust_ingest.process_exoplanet_data("data/KOI_Cumulative_clean.csv")
        df = df_polars.to_pandas()
        print("✅ Rust Ingestion Complete. Data processed in memory-safe dataframe.")
    except ImportError:
        print("⚠️ Rust module not found. Falling back to pandas (Make sure to build Docker image!)")
        df = pd.read_csv("data/KOI_Cumulative_clean.csv")
        cols_to_drop = ["rowid", "kepid", "kepoi_name", "kepler_name", "koi_disp_prov", "koi_comment", "koi_tce_delivname", "koi_fwm_stat_sig", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Minimal Preprocessing
    df = df.dropna(thresh=len(df)*0.5, axis=1) # Drop mostly empty columns
    
    # Fill remaining NaNs
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns
    for col in numeric_cols: df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols: df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['koi_disposition'])
    
    # Prepare Features
    X = df.drop(columns=['koi_disposition', 'koi_pdisposition'], errors='ignore')
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    print(f"🧠 Training Physics-Informed Neural Network (Features: {X.shape[1]})...")
    model = TabularAttentionNetwork(input_dim=X.shape[1], num_classes=len(label_encoder.classes_))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        logits, _ = model(X_train_t)
        loss = physics_informed_loss(logits, y_train_t, X_train_t, X.columns)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")
            
    print("✅ Training Complete.")
    
    # Evaluation and UMAP Latent Space Mapping
    print("🌌 Evaluating Model and Mapping Latent Space via UMAP...")
    model.eval()
    with torch.no_grad():
        logits, embeddings = model(X_test_t)
        predictions = torch.argmax(logits, dim=1).numpy()
        
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    print("\n--- 🏆 FINAL MODEL METRICS ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("------------------------------\n")
    
    # Fit UMAP on the neural network's internal representation
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings.numpy())
    
    # Create Interactive Plotly Scatter
    plot_df = pd.DataFrame({
        'UMAP_X': embedding_2d[:, 0],
        'UMAP_Y': embedding_2d[:, 1],
        'True_Label': label_encoder.inverse_transform(y_test),
        'Predicted_Label': label_encoder.inverse_transform(predictions)
    })
    
    os.makedirs("reports/figures", exist_ok=True)
    
    fig = px.scatter(plot_df, x='UMAP_X', y='UMAP_Y', color='True_Label', 
                     symbol='Predicted_Label', title='PINN Latent Space Embedding (UMAP)',
                     hover_data=['Predicted_Label'])
    fig.write_html("reports/figures/latent_space_map.html")
    print("📊 Latent Space Interactive Map saved to reports/figures/latent_space_map.html")

if __name__ == "__main__":
    main()
