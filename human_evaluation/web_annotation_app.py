import streamlit as st
from pathlib import Path
import csv
import os
# import streamlit.components.v1 as components # REMOVIDO

# --- Configuração Inicial ---
# Define o caminho base como o diretório atual
BASE_DATA_PATH = Path("batch_3") 
OUTPUT_FILE = Path("annotations_batch_3.csv")
NUM_COLUMNS = 6 # Quantas imagens por linha na grade (RE-ADICIONADO)

# --- Funções Auxiliares ---

@st.cache_data
def get_all_cod_folders(base_path):
    """Encontra recursivamente todas as pastas 'cod_XXX' dentro do caminho base, ignorando o .venv."""
    st.write(f"Procurando pastas em: {base_path.resolve()}")
    folders = sorted([p for p in base_path.glob("**/cod_*") if p.is_dir() and ".venv" not in str(p.resolve())])
    st.write(f"Encontradas {len(folders)} pastas de lotes.")
    return folders

def save_annotations(annotator, folder_path, all_images_in_folder): # REVERTIDO: Não precisa mais de 'selected_image_paths'
    """
    Salva as anotações para o lote atual.
    Usa um formato CSV "longo": uma linha por imagem.
    Formato: annotator, folder_path, image_name, is_good (1 ou 0)
    """
    file_exists = OUTPUT_FILE.is_file()
    
    try:
        with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Escreve o cabeçalho se o arquivo for novo
            if not file_exists:
                writer.writerow(["annotator", "folder_path", "image_name", "is_good"])
            
            # Itera por todas as imagens que foram mostradas na página
            for img_path in all_images_in_folder: # REVERTIDO: Itera pela lista de Path
                
                # REVERTIDO: Pega o valor do checkbox do session_state
                key = f"{folder_path.name}_{img_path.name}"
                is_good = 1 if st.session_state.get(key, False) else 0
                
                # Tenta salvar o caminho relativo ao diretório do app
                try:
                    relative_folder = str(img_path.parent.relative_to(BASE_DATA_PATH))
                except ValueError:
                    relative_folder = str(img_path.parent)

                writer.writerow([annotator, relative_folder, img_path.name, is_good])
                
    except PermissionError:
        st.error(f"Erro de Permissão: Não foi possível escrever em '{OUTPUT_FILE}'. Verifique se o arquivo não está aberto em outro programa (como o Excel).")
    except Exception as e:
        st.error(f"Ocorreu um erro ao salvar: {e}")

def display_instructions():
    """Modificação 2 e 3: Função reutilizável para mostrar as instruções."""
    st.write("Obrigado por se voluntariar! Você ajudará a construir um dataset de imagens tipográficas para o Museu do Ipiranga.")
    st.markdown("## Instruções - Leia com atenção")
    st.write("Para saber o que é uma boa imagem, vamos estabelecer alguns critérios.")

    st.subheader("São boas imagens:")
    
    # Função auxiliar para mostrar exemplo
    def show_example(text, path_str, width=150):
        st.write(f"- {text}")
        img_path = Path(path_str)
        if img_path.is_file():
            st.image(str(img_path), width=width)
        else:
            st.warning(f"Imagem de exemplo não encontrada: {path_str}")

    show_example("Um caracter íntegro perfeitamente recortado, sem invasões de outros caracteres ​✅​", "batch_1/escritural/cod_002/cod_002_000.png")
    show_example("Um caracter íntegro, com **pequenas** invasões de outros caracteres ​✅​", "batch_1/escritural/cod_026/cod_026_011.png")
    
        
    st.write("- Caracteres borrados, com borrões de tinta, ou desfocados, mas você reconhece que é um caracter ​✅")
    cols = st.columns(2)
    with cols[0]:
        show_example("'2' borrado", "batch_3/serifado/cod_158/cod_158_003.png", width=100)
    with cols[1]:
        show_example("'a' desfocado", "batch_2/escritural/cod_062/cod_062_013.png", width=100)
    st.divider() # ADICIONADO AQUI

    st.subheader("São imagens ruins:")
    show_example("Ruídos que não correspondem a nenhum caracter ❌​", "batch_1/escritural/cod_006/cod_006_018.png")
    
    st.write("- Partes soltas de caracteres (cabos, serifas, pingos de 'is'), que você não reconhece a qual ele pertence. Não ajuda a discernir característica alguma ❌​")
    cols = st.columns(2)
    with cols[0]:
        show_example("Fragmento de um caracter escritural", "batch_1/escritural/cod_006/cod_006_027.png", width=100)
    with cols[1]:
        show_example("Pingo de i", "batch_1/serifado/cod_020/cod_020_020.png", width=100)

    st.write("- Partes incompletas (fragmentos) de um caracter ​❌")
    cols = st.columns(2)
    with cols[0]:
        show_example("Fragmento de letra 'q'", "batch_1/serifado/cod_029/cod_029_007.png", width=100)
    with cols[1]:
        show_example("Fragmento do número '2'", "batch_1/serifado/cod_028/cod_028_026.png", width=100)
        
    show_example("Dois ou mais caracteres no mesmo recorte ​❌​", "batch_1/escritural/cod_057/cod_057_021.png")
    show_example("Linhas, pontos, e fragmentos sem conteúdo ​❌", "batch_1/escritural/cod_047/cod_047_143.png")


# --- Inicialização do Estado da Sessão ---
# O 'session_state' é como o Streamlit lembra das variáveis entre interações.

# 1. Carrega a lista de pastas (lotes) uma vez e guarda
if 'folders_to_process' not in st.session_state:
    st.session_state.folders_to_process = get_all_cod_folders(BASE_DATA_PATH)

# 2. Guarda o índice do lote (página) atual
if 'current_folder_index' not in st.session_state:
    st.session_state.current_folder_index = 0

# Modificação 2: Novo estado para controlar a tela de instruções
if 'show_instructions' not in st.session_state:
    st.session_state.show_instructions = True

# CORREÇÃO: Novo estado para controlar a tela de AJUDA
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# MODIFICAÇÃO: Novo estado para o nome do avaliador
if 'annotator_name' not in st.session_state:
    st.session_state.annotator_name = ""

# REMOVIDO: Novo estado para controlar o scroll
# if 'scroll_to_top' not in st.session_state:
#     st.session_state.scroll_to_top = False

# --- Interface Principal do App ---

st.set_page_config(layout="wide")

# REMOVIDO: Lógica para forçar o scroll para o topo
# if st.session_state.scroll_to_top:
#     ...

# REMOVIDO: Título movido para dentro das seções de "bloqueio"
# st.title("Ferrementa de Avaliação de Recortes 📷")

# --- 1. Identificação do Voluntário (MODIFICADO) ---
# Só pede o nome se ele AINDA NÃO foi definido.
if not st.session_state.annotator_name:
    st.title("Ferramenta de Avaliação de Recortes 📷") # ADICIONADO AQUI
    st.write("Aluno: Matheus Hencklein Ponte (m247277@dac.unicamp.br)")
    st.write("Orientadora: Prof. Dra. Paula Dornhofer Paro Costa (paulad@unicamp.br)")
    st.header("Bem-vindo(a)!")
    temp_name = st.text_input(
        "Insira seu Nome para começar:", 
        placeholder="Ex: voluntario_1"
    )
    
    if st.button("Confirmar Nome e Iniciar", type="primary"):
        if temp_name:
            st.session_state.annotator_name = temp_name
            st.rerun() # Recarrega a página, agora com o nome salvo
        else:
            st.warning("Por favor, insira seu nome para continuar.")
    
    st.stop() # Para a execução aqui ATÉ o nome ser confirmado

# REMOVIDO: Header do avaliador movido para dentro das seções de "bloqueio"
# st.header(f"Avaliador: {st.session_state.annotator_name}")
# st.divider()


# --- 2. Modificação 2: Tela de Instruções ---
if st.session_state.show_instructions:
    st.title("Ferramenta de Avaliação de Recortes 📷") # ADICIONADO AQUI
    st.header("Instruções de Avaliação")
    st.header(f"Avaliador: {st.session_state.annotator_name}") # ADICIONADO AQUI
    st.divider() # ADICIONADO AQUI
    display_instructions() # Chama a função com as instruções
    st.divider()
    
    # Botão para começar
    if st.button("Entendi, começar a avaliar!", type="primary"):
        st.session_state.show_instructions = False # "Fecha" a tela de instruções
        st.rerun() # Recarrega o script para ir para a tela de avaliação
    
    st.stop() # Para a execução aqui até o usuário clicar no botão

# CORREÇÃO: Lógica para exibir a tela de ajuda (substitui o modal/dialog)
if st.session_state.show_help:
    st.title("Ferramenta de Avaliação de Recortes 📷") # ADICIONADO AQUI
    st.header("Instruções de Avaliação")
    st.header(f"Avaliador: {st.session_state.annotator_name}") # ADICIONADO AQUI
    st.divider() # ADICIONADO AQUI
    
    # MODIFICADO: Botão "Fechar Ajuda" movido para o topo e alinhado à direita
    _, close_help_col = st.columns([0.8, 0.2]) 
    with close_help_col:
        if st.button("Voltar à avaliação", type="primary", use_container_width=True):
            st.session_state.show_help = False # "Fecha" a tela de ajuda
            st.rerun() # Recarrega o script para a tela de avaliação
    
    display_instructions() # Reutiliza a função de instruções
    st.divider()
    
    st.stop() # Para a execução aqui até o usuário clicar no botão

# --- 3. Verifica se o trabalho acabou ---
total_folders = len(st.session_state.folders_to_process)
current_index = st.session_state.current_folder_index

if current_index >= total_folders:
    st.success("🎉 Avaliação Concluída! 🎉")
    st.write("Todos os lotes foram processados. Muito obrigado pela sua ajuda!")
    st.header("Dúvidas ou Sugestões?")
    st.write("Contato: Matheus Hencklein Ponte (m247277@dac.unicamp.br)")
    
    st.balloons()
    st.stop()

# --- 4. Exibição do Lote Atual ---

# MOVIDO PARA CIMA: Define o lote atual e os arquivos ANTES dos botões
current_folder = st.session_state.folders_to_process[current_index]
image_files = sorted(list(current_folder.glob("*.png")))


# MODIFICADO: Botões de Ação (Ajuda e Próximo Lote)
with st.container():
    # Cria colunas: espaço vazio | botão de ajuda | botão de próximo
    _, help_col, next_col = st.columns([0.7, 0.1, 0.2]) 
    
    with help_col:
        if st.button("❓ Instruções"):
            # CORREÇÃO: Altera o estado para 'True' e recarrega a página
            st.session_state.show_help = True
            st.rerun()

    # MOVIDO: Lógica do botão "Salvar e Próximo Lote"
    with next_col:
        if st.button("Salvar e Próximo Lote", use_container_width=True, type="primary"):
            # 1. Salva as anotações da página atual
            save_annotations(st.session_state.annotator_name, current_folder, image_files)
            
            # 2. Avança o índice para o próximo lote
            st.session_state.current_folder_index += 1
            
            # 3. Força um "rerun" para recarregar a página com o novo índice.
            st.rerun()


# Barra de progresso
# MODIFICADO: Aumenta o tamanho da fonte do texto da barra de progresso
progress_text = f"Lote {current_index + 1} de {total_folders}"
st.markdown(f"<span style='font-size: 1.2rem; font-weight: 500;'>{progress_text}</span>", unsafe_allow_html=True)
st.progress((current_index + 1) / total_folders, text="") # Texto original removido para ser substituído pelo markdown

# st.header e st.caption usam 'current_folder' que já foi definido
st.header(f"Lote: {current_folder.relative_to(BASE_DATA_PATH)}")
st.caption(f"Caminho completo: {current_folder}")

# 'image_files' também já foi definido
if not image_files:
    st.warning("Nenhuma imagem .png encontrada nesta pasta.")
else:
    # REVERTIDO: Lógica original com checkboxes
    st.write("Selecione as imagens 'boas' marcando a caixa abaixo delas.")
    
    for i in range(0, len(image_files), NUM_COLUMNS):
        # Pega um "pedaço" da lista de imagens para esta linha
        row_images = image_files[i:i + NUM_COLUMNS]
        
        # Cria as colunas na interface
        cols = st.columns(NUM_COLUMNS)
        
        for col_index, img_path in enumerate(row_images):
            with cols[col_index]:
                # Chave única para o checkbox
                checkbox_key = f"{current_folder.name}_{img_path.name}"
                
                # Exibe a imagem
                st.image(
                    str(img_path), 
                    width="stretch", # MANTIDO: Alteração do usuário
                    caption=f"{img_path.name}"
                )
                
                # O st.checkbox armazena seu estado (True/False) no st.session_state
                # usando a 'key' fornecida.
                st.checkbox(
                    "Marcar como 'boa'", # Texto do checkbox
                    key=checkbox_key,
                )