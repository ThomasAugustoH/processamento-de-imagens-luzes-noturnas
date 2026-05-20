import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from preprocess_normalize import preprocess_image

def load_image(filepath):
    # Carregar imagem (aceita TIF e mantém suporte ao formato Float32 original do VIIRS)
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem: {filepath}")
    
    # Caso a imagem possua valores Not-a-Number (NaN) que correspondem a NoData, substitui por 0
    img = np.nan_to_num(img, nan=0.0)
    return img

def main():
    # Caminhos baseados na estrutura do seu workspace (Ex: Blumenau janeiro de 2015 e 2025)
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    img_2015_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "NTL_LITORAL", "NTL_2015", "VIIRS_NTL_MedianaMensal_Blumenau_2015_01_reprojetada.tif")
    img_2025_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "NTL_LITORAL", "NTL_2025", "VIIRS_NTL_MedianaMensal_Blumenau_2025_01_reprojetada.tif")
    
    print("Pré-processando e carregando imagens...")
    try:
        # Integrando com o módulo de pré-processamento.
        # Estamos pegando a imagem binarizada (apenas os focos reais de luz = 255 e fundo = 0)
        img_2015, _ = preprocess_image(img_2015_path, "blumenau_2015.png", threshold_value=15)
        img_2025, _ = preprocess_image(img_2025_path, "blumenau_2025.png", threshold_value=15)
    except Exception as e:
        print(f"Erro: {e}")
        return
        
    print(f"Dimensões da Imagem 2015: {img_2015.shape}")
    print(f"Dimensões da Imagem 2025: {img_2025.shape}")

    # Verificar se as imagens possuem dimensões compatíveis
    if img_2015.shape != img_2025.shape:
        print("Atenção: as imagens possuem tamanhos diferentes. Ajustando para o menor tamanho...")
        min_shape = (min(img_2015.shape[0], img_2025.shape[0]), min(img_2015.shape[1], img_2025.shape[1]))
        img_2015 = img_2015[:min_shape[0], :min_shape[1]]
        img_2025 = img_2025[:min_shape[0], :min_shape[1]]

    X = []
    y = []

    patch_size = 16  # Tamanho do bloco para extração de características
    threshold = 10.0  # Como a base agora é PB (0 e 255), um limiar de 10 na média significa que vários pixels "acenderam"

    print(f"Extraindo características em blocos de {patch_size}x{patch_size} pixels...")

    # Percorrer a imagem em blocos
    for i in range(0, img_2015.shape[0], patch_size):
        for j in range(0, img_2015.shape[1], patch_size):
            p1 = img_2015[i:i+patch_size, j:j+patch_size]
            p2 = img_2025[i:i+patch_size, j:j+patch_size]

            # Ignorar amostras cortadas nas bordas
            if p1.shape != (patch_size, patch_size):
                continue

            mean1 = np.mean(p1)
            mean2 = np.mean(p2)

            diff = mean2 - mean1

            features = [
                mean1,
                mean2,
                diff,
                np.var(p1),
                np.var(p2)
            ]

            # Rótulo = 1 (crescimento urbano) se a diferença superar o limiar, senão 0
            label = 1 if diff > threshold else 0

            X.append(features)
            y.append(label)

    print(f"Total de amostras extraídas: {len(X)}")
    
    # Modelo Random Forest
    print("Treinando classificador Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("Modelo treinado com sucesso!")
    
    # Criar uma imagem para visualizar o mapa de crescimento classificado
    print("Gerando mapa visual de crescimento...")
    crescimento_mapa = np.zeros(img_2015.shape, dtype=np.uint8)
    
    idx = 0
    for i in range(0, img_2015.shape[0], patch_size):
        for j in range(0, img_2015.shape[1], patch_size):
            p1 = img_2015[i:i+patch_size, j:j+patch_size]
            if p1.shape != (patch_size, patch_size):
                continue
            
            # Predição do modelo treinado
            label_pred = model.predict([X[idx]])[0]
            
            if label_pred == 1:
                # Marcar crescimento com branco (255) no mapa visual
                crescimento_mapa[i:i+patch_size, j:j+patch_size] = 255
            
            idx += 1

    # Salva o mapa de resultado como PNG
    output_path = os.path.join(base_path, "mapa_crescimento_blumenau.png")
    cv2.imwrite(output_path, crescimento_mapa)
    print(f"Mapa visual salvo em: {output_path}")
    print("Processo concluído.")

if __name__ == '__main__':
    main()
