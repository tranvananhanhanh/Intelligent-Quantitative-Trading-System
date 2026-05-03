#!/usr/bin/env python3
"""
🚀 UPDATE WEIGHTS: Tạo min-variance weights và lưu vào ml_weights_today.csv

Cách chạy:
  python src/strategies/update_weights_min_variance.py
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def update_weights_to_min_variance():
    """Cập nhật ml_weights_today.csv dùng min-variance weights"""
    
    print("\n" + "="*80)
    print("🚀 UPDATING ml_weights_today.csv WITH MIN-VARIANCE WEIGHTS")
    print("="*80)
    
    # Load fundamentals
    fund_path = PROJECT_ROOT / "data" / "fundamentals.csv"
    if not fund_path.exists():
        print(f"❌ ERROR: {fund_path} not found!")
        print("   Run Strategy Execution > Step 0 first to generate fundamentals.csv")
        return False
    
    print(f"\n📂 Loading fundamentals.csv...")
    fundamentals = pd.read_csv(fund_path)
    print(f"   ✅ Loaded: {len(fundamentals)} rows, {fundamentals['gvkey'].nunique()} stocks")
    
    # Initialize ML strategy
    print(f"\n🤖 Initializing ML Strategy...")
    try:
        from strategies.ml_strategy import MLStockSelectionStrategy
        from strategies.base_strategy import StrategyConfig
        
        config = StrategyConfig(
            name="Min-Variance Update",
            description="Auto-update weights to min-variance"
        )
        strategy = MLStockSelectionStrategy(config)
        print(f"   ✅ Strategy initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Generate min-variance weights
    print(f"\n📊 Generating min-variance weights...")
    try:
        result = strategy.generate_weights(
            data={"fundamentals": fundamentals},
            prediction_mode="single",
            test_quarters=4,
            top_quantile=0.9,
            weight_method="min_variance",  # ← KEY: min_variance!
            confirm_mode="today",
            execution_date=datetime.now().strftime("%Y-%m-%d"),
        )
        
        if result.weights.empty:
            print(f"   ❌ No stocks selected!")
            return False
        
        weights_df = result.weights[['gvkey', 'predicted_return', 'weight', 'date']].copy()
        weights_df = weights_df.sort_values('weight', ascending=False)
        
        print(f"   ✅ Generated {len(weights_df)} stocks")
        print(f"\n   📈 Weight Distribution:")
        print(f"      Min Weight:   {weights_df['weight'].min():.4f} ({weights_df['weight'].min()*100:.2f}%)")
        print(f"      Max Weight:   {weights_df['weight'].max():.4f} ({weights_df['weight'].max()*100:.2f}%)")
        print(f"      Mean Weight:  {weights_df['weight'].mean():.4f} ({weights_df['weight'].mean()*100:.2f}%)")
        print(f"      Std Weight:   {weights_df['weight'].std():.6f}")
        
        # Check if min_variance is working (weights should be varied)
        if weights_df['weight'].std() < 0.001:
            print(f"\n   ⚠️  WARNING: Weights are still equal (std = {weights_df['weight'].std():.6f})")
            print(f"      Min-variance may not be working properly")
        else:
            print(f"\n   ✅ Min-variance is WORKING!")
            print(f"      Weights are varied (std = {weights_df['weight'].std():.6f})")
        
    except Exception as e:
        print(f"   ❌ Error generating weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save to CSV
    print(f"\n💾 Saving to ml_weights_today.csv...")
    out_path = PROJECT_ROOT / "data" / "ml_weights_today.csv"
    try:
        weights_df.to_csv(out_path, index=False)
        print(f"   ✅ Saved to: {out_path}")
    except Exception as e:
        print(f"   ❌ Error saving: {e}")
        return False
    
    # Display results
    print(f"\n📋 Top 10 Stocks (sorted by weight):")
    print(weights_df[['gvkey', 'weight', 'predicted_return']].head(10).to_string(index=False))
    
    # Verify file was updated
    print(f"\n✅ VERIFICATION:")
    verify_df = pd.read_csv(out_path)
    print(f"   Stocks: {len(verify_df)}")
    print(f"   Weight range: {verify_df['weight'].min():.4f} - {verify_df['weight'].max():.4f}")
    print(f"   Weight std: {verify_df['weight'].std():.6f}")
    
    if verify_df['weight'].std() > 0.001:
        print(f"\n🎉 SUCCESS! Min-variance weights saved!")
        return True
    else:
        print(f"\n⚠️  File saved but weights appear equal...")
        return False


if __name__ == "__main__":
    success = update_weights_to_min_variance()
    
    if success:
        print("\n" + "="*80)
        print("✅ WEIGHTS UPDATED! Run test to verify:")
        print("   python test_min_variance.py")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ UPDATE FAILED!")
        print("="*80 + "\n")
        sys.exit(1)
