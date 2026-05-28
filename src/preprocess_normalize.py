import cv2
import numpy as np
import os

def preprocess_image(input_data, output_filename, threshold_value=None):
    # Carregar imagem a partir do caminho (str) ou aceitar o array numpy diretamente
    if isinstance(input_data, str):
        img = cv2.imread(input_data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Erro ao carregar a imagem: {input_data}")
    else:
        img = input_data.copy()
    
    # Substituir Not-A-Number (NaN) por 0
    img = np.nan_to_num(img, nan=0.0)
    
    # Garantir que não existam valores negativos (ruído indesejado de sensores de satélite)
    img[img < 0] = 0
    
    # Remove outliers extremos cortando valores acima do percentil 99.9%
    max_val = np.percentile(img, 99.9)
    img_clipped = np.clip(img, 0, max_val)

    # 1. NORMALIZAÇÃO:
    img_norm = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
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
    output_dir = os.path.join(base_dir, "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    
    norm_path = os.path.join(output_dir, f"normalizada_{output_filename}")
    bin_path = os.path.join(output_dir, f"binaria_{output_filename}")
    
    cv2.imwrite(norm_path, img_norm)
    cv2.imwrite(bin_path, img_binaria)
    
    print(f"Arquivos salvos:\n -> {norm_path}\n -> {bin_path}")
    return img_binaria, img_norm

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    print("Iniciando Módulo de Pré-processamento e Normalização...")
    
    for year in range(2015, 2026):
        images_of_year = []
        
        # Iterar por todos os meses (1 a 12)
        for month in range(1, 13):
            month_str = f"{month:02d}"
            img_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "NTL_LITORAL", f"NTL_{year}", f"VIIRS_NTL_MedianaMensal_Blumenau_{year}_{month_str}_reprojetada.tif")
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images_of_year.append(img)

        if images_of_year:
            print(f"\nProcessando Blumenau {year} ({len(images_of_year)} meses encontrados)...")
            
            # Garante que tenham a mesma dimensão antes do empilhamento
            min_h = min(im.shape[0] for im in images_of_year)
            min_w = min(im.shape[1] for im in images_of_year)
            images_cropped = [im[:min_h, :min_w] for im in images_of_year]
            
            # Calcula a mediana de todos os meses para cada pixel
            median_img = np.median(np.stack(images_cropped, axis=0), axis=0)
            
            preprocess_image(median_img, f"blumenau_{year}.png", threshold_value=15)

if __name__ == '__main__':
    main()
