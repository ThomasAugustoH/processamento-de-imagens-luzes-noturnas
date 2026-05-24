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
    
    print("Pré-processando e carregando imagens...")
    available_images = {}
    for year in range(2015, 2026):
        img_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "NTL_LITORAL", f"NTL_{year}", f"VIIRS_NTL_MedianaMensal_Blumenau_{year}_01_reprojetada.tif")
        if os.path.exists(img_path):
            try:
                print(f"Processando imagem do ano {year}...")
                img, _ = preprocess_image(img_path, f"blumenau_{year}.png", threshold_value=15)
                available_images[year] = img
            except Exception as e:
                print(f"Erro ao carregar o ano {year}: {e}")
                
    if len(available_images) < 2:
        print("Erro: É necessário ter pelo menos dois anos de imagens para treinar o modelo.")
        return

    # Caminho para o modelo digital de elevação (DEM). Ex: Arquivo SRTM ou NASADEM 
    dem_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "DEM_Blumenau_reprojetada.tif")
    
    try:
        # Carrega o DEM (sem binarizar para manter a altitude). Usa zeros se o arquivo ainda não existir (Mock).
        if os.path.exists(dem_path):
            img_dem = load_image(dem_path)
        else:
            print("Aviso: DEM não encontrado. Usando altitude zero (mock) para permitir a execução.")
            img_dem = np.zeros_like(list(available_images.values())[0])
    except Exception as e:
        print(f"Erro ao carregar DEM: {e}")
        return
        
    years_list = sorted(available_images.keys())
    print(f"Anos carregados com sucesso: {years_list}")

    # Verificar se as imagens possuem dimensões compatíveis e ajustar
    shapes = [img.shape for img in available_images.values()] + [img_dem.shape]
    min_shape = (min(s[0] for s in shapes), min(s[1] for s in shapes))
    print(f"Ajustando imagens para o tamanho comum: {min_shape}")
    
    for year in available_images:
        available_images[year] = available_images[year][:min_shape[0], :min_shape[1]]
    img_dem = img_dem[:min_shape[0], :min_shape[1]]

    X_train = []
    y_train = []

    patch_size = 8  # Alterado para 8 para aumentar a resolução
    threshold_base = 10.0  # Limiar base de crescimento considerando um gap de 10 anos

    print(f"Extraindo características ao longo do tempo em blocos de {patch_size}x{patch_size} pixels...")

    # Percorrer os anos disponíveis aos pares (ex: 2015->2016, 2016->2019)
    for idx_year in range(len(years_list) - 1):
        y1 = years_list[idx_year]
        y2 = years_list[idx_year + 1]
        delta_years = y2 - y1
        
        img1 = available_images[y1]
        img2 = available_images[y2]
        
        # Ajusta o limiar de crescimento esperado proporcionalmente ao tempo decorrido
        threshold = (threshold_base / 10.0) * delta_years

        for i in range(0, img1.shape[0], patch_size):
            for j in range(0, img1.shape[1], patch_size):
                p1 = img1[i:i+patch_size, j:j+patch_size]
                p2 = img2[i:i+patch_size, j:j+patch_size]
                p_dem = img_dem[i:i+patch_size, j:j+patch_size]

                # Ignorar amostras cortadas nas bordas
                if p1.shape != (patch_size, patch_size):
                    continue

                mean1 = np.mean(p1)
                mean2 = np.mean(p2)
                mean_dem = np.mean(p_dem)
                var_dem = np.var(p_dem)

                diff = mean2 - mean1

                # Features de TREINAMENTO: Luz inicial + Topografia + Tempo
                features_train = [
                    mean1,
                    np.var(p1),
                    mean_dem,
                    var_dem,
                    delta_years  # Nova feature: informa ao modelo quanto tempo passou
                ]
                
                label = 1 if diff > threshold else 0

                X_train.append(features_train)
                y_train.append(label)

    print(f"Total de amostras de treino extraídas: {len(X_train)}")
    
    # Modelo Random Forest
    print("Treinando classificador Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    
    print("Gerando mapa de previsão de crescimento FUTURO...")
    last_year = years_list[-1]
    img_last = available_images[last_year]
    crescimento_mapa = np.zeros(img_last.shape, dtype=np.uint8)
    
    future_delta_years = 10  # Prever os próximos 10 anos
    
    for i in range(0, img_last.shape[0], patch_size):
        for j in range(0, img_last.shape[1], patch_size):
            p_last = img_last[i:i+patch_size, j:j+patch_size]
            p_dem = img_dem[i:i+patch_size, j:j+patch_size]
            
            if p_last.shape != (patch_size, patch_size):
                continue
            
            # Features futuras baseadas no último ano disponível
            features_future = [
                np.mean(p_last),
                np.var(p_last),
                np.mean(p_dem),
                np.var(p_dem),
                future_delta_years
            ]
            
            label_pred = model.predict([features_future])[0]
            
            if label_pred == 1:
                # Marcar crescimento com branco (255) no mapa visual
                crescimento_mapa[i:i+patch_size, j:j+patch_size] = 255
            
    # --- PÓS-PROCESSAMENTO PARA REMOVER O EFEITO BLOCKY ---
    print("Aplicando suavização morfológica no mapa gerado...")
    kernel = np.ones((5, 5), np.uint8)
    # 1. Closing: junta blocos que estão muito próximos (preenche fendas)
    crescimento_mapa = cv2.morphologyEx(crescimento_mapa, cv2.MORPH_CLOSE, kernel)
    # 2. Gaussian Blur: desfoca as bordas quadradas duras dos blocos
    mapa_blur = cv2.GaussianBlur(crescimento_mapa, (11, 11), 0)
    # 3. Threshold: endurece novamente o desfoque, criando bordas arredondadas e orgânicas
    _, crescimento_mapa_suave = cv2.threshold(mapa_blur, 127, 255, cv2.THRESH_BINARY)

    output_dir = os.path.join(base_path, "generated images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mapa_previsao_futura_blumenau.png")
    cv2.imwrite(output_path, crescimento_mapa_suave)
    print(f"Mapa visual salvo em: {output_path}")
    print("Processo concluído.")

if __name__ == '__main__':
    main()
