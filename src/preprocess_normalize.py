import cv2
import numpy as np
import os

def preprocess_image(filepath, output_filename, threshold_value=None):
    # Carregar imagem
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem: {filepath}")
    
    # Substituir Not-A-Number (NaN) por 0
    img = np.nan_to_num(img, nan=0.0)
    
    # Garantir que não existam valores negativos (ruído indesejado de sensores de satélite)
    img[img < 0] = 0
    
    # 1. NORMALIZAÇÃO:
    # A imagem TIF bruta de NTL geralmente tem ranges imensos. 
    # Aqui vamos normalizar todos os valores proporcialmente para a escala padrão de imagem (0 até 255)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 2. BINARIZAÇÃO (Transfomação em Preto e Branco puro):
    # Todos os pixels acima de 'threshold_value' viram branco (255), os que estão abaixo viram preto (0)
    if threshold_value is None:
        # Se não informarmos limite, usamos o algoritmo de Otsu que calcula o ponto ótimo matematicamente
        ret, img_binaria = cv2.threshold(img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Limiar de Otsu calculado: {ret}")
    else:
        # Usamos o limite manual que escolhermos (0 a 255)
        ret, img_binaria = cv2.threshold(img_norm, threshold_value, 255, cv2.THRESH_BINARY)
        print(f"Limiar manual utilizado: {threshold_value}")
        
    # Salvar a imagem normalizada em cinza e a imagem binarizada PB
    base_dir = os.path.dirname(os.path.abspath(__file__))
    norm_path = os.path.join(base_dir, f"normalizada_{output_filename}")
    bin_path = os.path.join(base_dir, f"binaria_{output_filename}")
    
    cv2.imwrite(norm_path, img_norm)
    cv2.imwrite(bin_path, img_binaria)
    
    print(f"Arquivos salvos:\n -> {norm_path}\n -> {bin_path}")
    return img_binaria, img_norm

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Aproveitando os caminhos já trabalhados pra Blumenau
    img_2015_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "NTL_LITORAL", "NTL_2015", "VIIRS_NTL_MedianaMensal_Blumenau_2015_01_reprojetada.tif")
    img_2025_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "NTL_LITORAL", "NTL_2025", "VIIRS_NTL_MedianaMensal_Blumenau_2025_01_reprojetada.tif")
    
    print("Iniciando Módulo de Pré-processamento e Normalização...")
    
    # Processa 2015
    print("\nProcessando Blumenau 2015...")
    # Estamos testando com limite = 15 na escala de 0 a 255. 
    # Todo pixel com brilho > 15 acende, abaixo disso some (apagando o ruído)
    preprocess_image(img_2015_path, "blumenau_2015.png", threshold_value=15)
    
    # Processa 2025
    print("\nProcessando Blumenau 2025...")
    preprocess_image(img_2025_path, "blumenau_2025.png", threshold_value=15)

if __name__ == '__main__':
    main()
