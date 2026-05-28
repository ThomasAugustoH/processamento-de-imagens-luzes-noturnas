import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, jaccard_score
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
        images_of_year = []
        
        for month in range(1, 13):
            month_str = f"{month:02d}"
            img_path = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "NTL_LITORAL", f"NTL_{year}", f"VIIRS_NTL_MedianaMensal_Blumenau_{year}_{month_str}_reprojetada.tif")
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    images_of_year.append(img)
                    
        if images_of_year:
            try:
                min_h = min(im.shape[0] for im in images_of_year)
                min_w = min(im.shape[1] for im in images_of_year)
                images_cropped = [im[:min_h, :min_w] for im in images_of_year]
                
                print(f"Processando imagem mediana do ano {year} ({len(images_of_year)} meses encontrados)...")
                median_img = np.median(np.stack(images_cropped, axis=0), axis=0)
                
                img, _ = preprocess_image(median_img, f"blumenau_{year}.png", threshold_value=15)
                available_images[year] = img
            except Exception as e:
                print(f"Erro ao processar o ano {year}: {e}")
                
    if len(available_images) < 2:
        print("Erro: É necessário ter pelo menos dois anos de imagens para treinar o modelo.")
        return

    # Busca dinâmica pelo modelo digital de elevação (DEM) na pasta RASTER
    raster_dir = os.path.join(base_path, "data", "NTL_LITORAL_SC", "QGIS_LITORAL", "RASTER", "MDE")
    dem_path = None
    
    if os.path.exists(raster_dir):
        for filename in os.listdir(raster_dir):
            nome_min = filename.lower()
            # Procura por arquivos raster que contenham dem, srtm ou elevation no nome
            if (nome_min.endswith(".tif") or nome_min.endswith(".tiff")) and ("dem" in nome_min or "srtm" in nome_min or "elevation" in nome_min):
                dem_path = os.path.join(raster_dir, filename)
                print(f"Arquivo DEM encontrado automaticamente: {filename}")
                break

    try:
        # Carrega o DEM (sem binarizar para manter a altitude). Usa zeros se o arquivo ainda não existir (Mock).
        if dem_path and os.path.exists(dem_path):
            img_dem = load_image(dem_path)
            print("DEM carregado com sucesso para o treinamento.")
        else:
            print("Aviso: Nenhum arquivo DEM encontrado na pasta RASTER. Usando altitude zero (mock) para permitir a execução.")
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

    X_train_val = []
    y_train_val = []
    X_train_all = []
    y_train_all = []

    ksize = 7  # Tamanho da janela local para extração de características (substitui blocos)
    step = 2   # Aumenta a densidade de amostras (4x mais dados) para o modelo aprender melhor as bordas
    threshold_base = 5.0  # Limiar mais rigoroso para ignorar ruídos e focar em crescimento real

    print(f"Extraindo características ao longo do tempo usando janelas locais de {ksize}x{ksize} pixels...")

    # Pre-calcular features do DEM para a imagem toda de uma vez
    dem_float = img_dem.astype(np.float32)
    mean_dem = cv2.blur(dem_float, (ksize, ksize))
    mean_sq_dem = cv2.blur(dem_float**2, (ksize, ksize))
    var_dem = np.maximum(mean_sq_dem - mean_dem**2, 0)

    # Nova Feature: Declividade (Slope) calculada via Gradiente de Sobel no DEM
    sobelx = cv2.Sobel(dem_float, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(dem_float, cv2.CV_32F, 0, 1, ksize=3)
    slope_dem = cv2.magnitude(sobelx, sobely)
    slope_blur = cv2.blur(slope_dem, (ksize, ksize))

    # Percorrer os anos disponíveis aos pares (ex: 2015->2016, 2016->2019)
    for idx_year in range(len(years_list) - 1):
        y1 = years_list[idx_year]
        y2 = years_list[idx_year + 1]
        delta_years = y2 - y1
        
        img1 = available_images[y1].astype(np.float32)
        img2 = available_images[y2].astype(np.float32)
        
        # Ajusta o limiar de crescimento esperado proporcionalmente ao tempo decorrido
        threshold = (threshold_base / 10.0) * delta_years

        # Extração vetorizada rápida de médias e variâncias locais
        mean1 = cv2.blur(img1, (ksize, ksize))
        mean_sq1 = cv2.blur(img1**2, (ksize, ksize))
        var1 = np.maximum(mean_sq1 - mean1**2, 0)
        
        # Feature: Contexto regional da iluminação (ajustado para 11x11 para não diluir demais)
        macro_mean1 = cv2.blur(img1, (11, 11))
        
        mean2 = cv2.blur(img2, (ksize, ksize))
        diff = mean2 - mean1

        # Amostragem para treinamento selecionando pixels intercalados (step)
        m1_sample = mean1[::step, ::step].flatten()
        v1_sample = var1[::step, ::step].flatten()
        macro_sample = macro_mean1[::step, ::step].flatten()
        md_sample = mean_dem[::step, ::step].flatten()
        vd_sample = var_dem[::step, ::step].flatten()
        sd_sample = slope_blur[::step, ::step].flatten()
        # Nova Feature: Diferença entre o pixel e seu bairro (ajuda a detectar bordas de loteamentos)
        local_macro_diff = m1_sample - macro_sample
        df_sample = diff[::step, ::step].flatten()
        
        delta_array = np.full_like(m1_sample, delta_years)
        
        features_matrix = np.column_stack((m1_sample, v1_sample, macro_sample, local_macro_diff, md_sample, vd_sample, sd_sample, delta_array))
        labels_array = (df_sample > threshold).astype(int)

        # Separa os dados de treinamento exclusivos para a etapa de validação (treinar até 2020)
        if y2 <= 2020:
            X_train_val.extend(features_matrix.tolist())
            y_train_val.extend(labels_array.tolist())
            
        # Acumula também para o modelo final que usa todos os dados
        X_train_all.extend(features_matrix.tolist())
        y_train_all.extend(labels_array.tolist())

    print(f"Total de amostras de treino extraídas (até 2020): {len(X_train_val)}")
    print(f"Total de amostras de treino extraídas (todos os anos): {len(X_train_all)}")
    
    # --- ETAPA DE VALIDAÇÃO (2020 -> 2024) ---
    if len(X_train_val) > 0 and 2020 in available_images and 2024 in available_images:
        print("Treinando modelo para validação (dados até 2020)...")
        model_val = RandomForestClassifier(n_estimators=1500, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42)
        model_val.fit(X_train_val, y_train_val)
        
        print("Validando modelo (prevendo 2024 a partir de 2020)...")
        img2020 = available_images[2020].astype(np.float32)
        img2024 = available_images[2024].astype(np.float32)
        
        delta_val = 4
        threshold_val = (threshold_base / 10.0) * delta_val
        
        # Características de 2020 reais e o Target verdadeiro de 2024
        mean2020 = cv2.blur(img2020, (ksize, ksize))
        var2020 = np.maximum(cv2.blur(img2020**2, (ksize, ksize)) - mean2020**2, 0)
        macro2020 = cv2.blur(img2020, (11, 11))
        diff_val = cv2.blur(img2024, (ksize, ksize)) - mean2020
        
        y_true_val = (diff_val > threshold_val).astype(int).flatten()
        local_macro_diff_val = mean2020.flatten() - macro2020.flatten()
        X_val = np.column_stack((mean2020.flatten(), var2020.flatten(), macro2020.flatten(), local_macro_diff_val,
                                 mean_dem.flatten(), var_dem.flatten(), slope_blur.flatten(), 
                                 np.full_like(mean2020.flatten(), delta_val)))
        
        y_pred_val = model_val.predict(X_val)
        f1 = f1_score(y_true_val, y_pred_val, zero_division=0)
        iou = jaccard_score(y_true_val, y_pred_val)
        
        print("\n" + "="*50)
        print(" RESULTADOS DA VALIDAÇÃO (2020 -> 2024)")
        print(f" F1-Score : {f1:.4f}")
        print(f" IoU      : {iou:.4f}")
        print("="*50 + "\n")
    else:
        print("Aviso: Dados insuficientes para validação 2020->2024 (verifique os anos disponíveis).")

    # Modelo Random Forest (adicionado class_weight='balanced' para lidar com o desbalanceamento)
    print("Treinando classificador Random Forest final com todos os dados...")
    model = RandomForestClassifier(n_estimators=500, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=42)
    model.fit(X_train_all, y_train_all)
    print("Modelo treinado com sucesso!")
    
    print("Gerando mapa de previsão de crescimento FUTURO (pixel a pixel)...")
    last_year = years_list[-1]
    img_last = available_images[last_year].astype(np.float32)
    
    mean_last = cv2.blur(img_last, (ksize, ksize))
    mean_sq_last = cv2.blur(img_last**2, (ksize, ksize))
    var_last = np.maximum(mean_sq_last - mean_last**2, 0)
    macro_last = cv2.blur(img_last, (11, 11))
    
    future_delta_years = 10  # Prever os próximos 10 anos
    
    m_last_flat = mean_last.flatten()
    v_last_flat = var_last.flatten()
    macro_last_flat = macro_last.flatten()
    local_macro_diff_last = m_last_flat - macro_last_flat
    md_flat = mean_dem.flatten()
    vd_flat = var_dem.flatten()
    sd_flat = slope_blur.flatten()
    delta_future_array = np.full_like(m_last_flat, future_delta_years)
    
    X_future = np.column_stack((m_last_flat, v_last_flat, macro_last_flat, local_macro_diff_last, md_flat, vd_flat, sd_flat, delta_future_array))
    
    print("Realizando predições no mapa completo...")
    preds = model.predict(X_future)
    
    crescimento_mapa = preds.reshape(img_last.shape).astype(np.uint8) * 255
            
    # --- PÓS-PROCESSAMENTO PARA REMOVER O EFEITO BLOCKY ---
    # Com a predição pixel a pixel, a morfologia atua apenas para refinar ruídos suaves.
    print("Aplicando suavização morfológica no mapa gerado...")
    kernel = np.ones((3, 3), np.uint8)
    # 1. Opening: Remove pixels isolados (ruído do tipo salt-and-pepper)
    crescimento_mapa = cv2.morphologyEx(crescimento_mapa, cv2.MORPH_OPEN, kernel)
    # 2. Closing: junta áreas que estão muito próximas (preenche pequenas fendas)
    crescimento_mapa = cv2.morphologyEx(crescimento_mapa, cv2.MORPH_CLOSE, kernel)
    
    # 3. Gaussian Blur: desfoca as bordas levemente
    mapa_blur = cv2.GaussianBlur(crescimento_mapa, (5, 5), 0)
    # 4. Threshold: endurece novamente o desfoque, criando bordas contínuas e orgânicas
    _, crescimento_mapa_suave = cv2.threshold(mapa_blur, 127, 255, cv2.THRESH_BINARY)

    output_dir = os.path.join(base_path, "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mapa_previsao_futura_blumenau.png")
    cv2.imwrite(output_path, crescimento_mapa_suave)
    print(f"Mapa visual salvo em: {output_path}")

    # --- GERAR MAPA COMBINADO (CIDADE EXISTENTE + CRESCIMENTO) ---
    mapa_combinado = np.zeros_like(img_last, dtype=np.uint8)
    # 1. Pinta a cidade existente do último ano de cinza (127)
    mapa_combinado[img_last > 0] = 127
    # 2. Pinta o novo crescimento previsto de branco (255)
    mapa_combinado[crescimento_mapa_suave > 0] = 255
    
    output_path_combinado = os.path.join(output_dir, "mapa_previsao_futura_combinado.png")
    cv2.imwrite(output_path_combinado, mapa_combinado)
    print(f"Mapa combinado salvo em: {output_path_combinado}")

    print("Processo concluído.")

if __name__ == '__main__':
    main()
