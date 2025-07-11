import pandas as pd
from itertools import combinations
from collections import defaultdict

# Cargar el dataset
df = pd.read_csv("OnlineRetail_Cleaned.csv")

# Eliminar filas con valores nulos en 'Invoice' o 'StockCode'
df.dropna(subset=['InvoiceNo', 'StockCode'], inplace=True)

# Agrupar los productos por cada transacción (Invoice)
invoice_products = df.groupby('InvoiceNo')['StockCode'].apply(set)

# Inicializar diccionario de coocurrencias
cooccurrence = defaultdict(int)

# Contar coocurrencias
for products in invoice_products:
    for prod1, prod2 in combinations(products, 2):
        pair = tuple(sorted((prod1, prod2)))
        cooccurrence[pair] += 1

print(f"Total de pares únicos de productos: {len(cooccurrence)}")

print("starting to build the affinity matrix...")

# Obtener todos los productos únicos
products = sorted(df['StockCode'].unique())

# Crear DataFrame vacío para la matriz
affinity_matrix = pd.DataFrame(0, index=products, columns=products)

# Llenar la matriz con los valores de coocurrencia
for (prod1, prod2), count in cooccurrence.items():
    affinity_matrix.at[prod1, prod2] = count
    affinity_matrix.at[prod2, prod1] = count  # simétrica

# Diagonal con ceros (un producto no tiene afinidad consigo mismo)
for prod in products:
    affinity_matrix.at[prod, prod] = 0

# Guardar la matriz de afinidad en un archivo CSV
affinity_matrix.to_csv("affinity_matrix.csv")

# Mostrar una parte de la matriz
print(affinity_matrix.iloc[:10, :10])  # Mostrar primeras 10 filas y columnas

"""invoice_id = '489434'
product_code = '85048'

# Verificar si existe esa factura en los datos
if invoice_id in invoice_products:
    if product_code in invoice_products[invoice_id]:
        print(f"✅ El producto {product_code} está en la factura {invoice_id}.")
    else:
        print(f"❌ El producto {product_code} NO está en la factura {invoice_id}.")
else:
    print(f"⚠️ La factura {invoice_id} no se encuentra en los datos.")"""