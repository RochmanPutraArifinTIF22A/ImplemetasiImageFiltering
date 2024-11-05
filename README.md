# ImplemetasiImageFiltering
import numpy as np

# Langkah 1: Buat matriks kernel (filter)
H = np.array([[1, 1, 1],   # Contoh kernel deteksi tepi (edge detection)
              [1, 4, 1],
              [1, 1, 1]])

# Langkah 2: Buat matriks gambar input (image original)
X = np.array([[1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0],
              [1, 1, 1, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]])

# Langkah 3: Tentukan fungsi konvolusi dengan parameter stride dan padding
def convolve2d(X, H, stride=1, padding=0):
    # Menambahkan padding pada gambar input (X)
    if padding > 0:
        X = np.pad(X, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    
    # Ukuran gambar input setelah padding
    x_height, x_width = X.shape
    h_height, h_width = H.shape
    
    # Ukuran output berdasarkan ukuran input dan kernel
    output_height = (x_height - h_height) // stride + 1
    output_width = (x_width - h_width) // stride + 1
    Y = np.zeros((output_height, output_width))
    
    # Melakukan konvolusi
    for i in range(0, output_height * stride, stride):
        for j in range(0, output_width * stride, stride):
            # Ambil sub-matriks dari input yang sesuai dengan kernel
            region = X[i:i+h_height, j:j+h_width]
            # Lakukan perkalian elemen-wise dan jumlahkan
            Y[i // stride, j // stride] = np.sum(region * H)
    
    return Y

# Langkah 4: Lakukan konvolusi pada gambar input dengan kernel
Y = convolve2d(X, H, stride=1, padding=0)

# Tampilkan hasil
print("Gambar Input (X):")
print(X)

print("\nKernel (H):")
print(H)

print("\nGambar Output (Y) setelah konvolusi:")
print(Y)
