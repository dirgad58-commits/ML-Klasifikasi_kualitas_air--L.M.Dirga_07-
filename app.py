# ... (semua kode training hingga bagian 9) ...

print("\n" + "="*60)
print("PROGRAM SELESAI. MENGGUNAKAN ADASYN + GRID SEARCH (parameter efisien).")
print("="*60)

# ==============================================
# 10. MENYIMPAN MODEL DAN ARTEFAK UNTUK DEPLOYMENT
# ==============================================
print("\n" + "="*60)
print("10. MENYIMPAN MODEL DAN ARTEFAK UNTUK DEPLOYMENT")
print("="*60)

import pickle
import json
import os

deploy_folder = 'deployment_files'
os.makedirs(deploy_folder, exist_ok=True)

for name, model in best_models.items():
    filename = name.lower().replace(' ', '_') + '.pkl'
    with open(os.path.join(deploy_folder, filename), 'wb') as f:
        pickle.dump(model, f)
    print(f"Model {name} disimpan ke {filename}")

# ... (sisa kode penyimpanan)
