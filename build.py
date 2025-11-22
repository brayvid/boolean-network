from app import generate_rbn_visuals

print("------------------------------------------------")
print("BUILD STEP: Pre-generating default RBN media...")
print("------------------------------------------------")

# Force generation for the default case (Node 6, Seed 42)
# The logic in app.py handles saving this to disk.
result = generate_rbn_visuals(6, 42)

if result.get('error_message'):
    print(f"❌ Build Failed: {result['error_message']}")
    exit(1)
else:
    print("✅ Default media generated successfully.")
    print(f"   Heatmap: {result['heatmap_data']}")
    print(f"   Animation: {result['animation_data']}")