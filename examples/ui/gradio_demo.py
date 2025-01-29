import asyncio
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    import gradio as gr
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from browser_use import Agent

    print("Successfully imported all required packages")
except Exception as e:
    print(f"Error importing packages: {str(e)}")
    sys.exit(1)

try:
    load_dotenv()
    print("Environment variables loaded")
except Exception as e:
    print(f"Error loading environment variables: {str(e)}")
    sys.exit(1)

@dataclass
class ActionResult:
    is_done: bool
    extracted_content: Optional[str]
    error: Optional[str]
    include_in_memory: bool

@dataclass
class AgentHistoryList:
    all_results: List[ActionResult]
    all_model_outputs: List[dict]

def format_result(result_str: str) -> str:
    try:
        if "AgentHistoryList" in result_str:
            if "extracted_content='" in result_str:
                content = result_str.split("extracted_content='")[-1].split("',")[0]
                if content:
                    lines = content.split('\\n')
                    formatted_lines = []
                    
                    # Check if this is a YouTube trending videos result
                    is_youtube_trending = any('görüntüleme' in line.lower() for line in lines)
                    
                    if is_youtube_trending:
                        formatted_lines.append("📄 Result: Here are the titles and view counts for the top 5 trending YouTube videos:\n")
                        video_count = 1
                        for line in lines:
                            if 'görüntüleme' in line.lower():
                                # Split title and views if they're on the same line
                                parts = line.split(' - ')
                                if len(parts) == 2:
                                    title, views = parts
                                    formatted_lines.append(f"{video_count}. **{title.strip()}** - {views.strip()}\n")
                                    video_count += 1
                    else:
                        for line in lines:
                            if line.strip():
                                if any(keyword in line.lower() for keyword in ['hata', 'error']):
                                    line = f"❌ {line}"
                                elif line[0].isdigit() and '.' in line:
                                    line = f"🔍 {line}"
                                else:
                                    line = f"📌 {line}"
                                formatted_lines.append(line)
                        formatted_lines.insert(0, "### 🎯 Sonuçlar\n")
                    
                    return "\n".join(formatted_lines)
            
            if "error='" in result_str:
                error = result_str.split("error='")[-1].split("',")[0]
                if error and error != "None":
                    return f"### ❌ Hata\n\n{error}"
        
        return result_str
    except Exception as e:
        print(f"Error formatting result: {str(e)}")
        return result_str

async def run_browser_task(
    task: str,
    headless: bool = True,
    model_choice: str = 'GPT-4',
    max_steps: int = 10
) -> str:
    try:
        print(f"Starting task: {task}")
        print(f"Model: {model_choice}, Headless: {headless}, Max steps: {max_steps}")
        
        if model_choice == 'GPT-4':
            model = "gpt-4o"
        else:
            model = "gpt-3.5-turbo"
        
        openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_key:
            return "### ❌ Hata\n\nOpenRouter API anahtarı bulunamadı!"
            
        llm = ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )
        
        agent = Agent(
            task=task,
            llm=llm
        )
        result = await agent.run(max_steps=max_steps)
        print("Task completed successfully")
        return format_result(str(result))
    except Exception as e:
        print(f"Error running task: {str(e)}")
        return f'### ❌ Hata\n\n{str(e)}'

def create_ui():
    print("Creating Gradio interface...")
    
    with gr.Blocks(
        title='Browser Use - Web Otomasyon Aracı',
        theme=gr.themes.Soft(),
        css=".gradio-container {max-width: 1200px !important}"
    ) as interface:
        with gr.Row():
            gr.Markdown('# 🤖 Web Otomasyon Asistanı')
        
        with gr.Row():
            gr.Markdown('### AI destekli web otomasyonu için akıllı arayüz')
        
        with gr.Tabs():
            with gr.TabItem("🎯 Görev"):
                with gr.Row():
                    with gr.Column(scale=2):
                        task = gr.Textbox(
                            label='Görev Açıklaması',
                            placeholder='Örnek: Amazon\'da laptop ara, en iyi puanlılara göre sırala ve ilk ürünün fiyatını göster',
                            lines=3
                        )
                        
                        with gr.Row():
                            with gr.Column():
                                model_choice = gr.Radio(
                                    choices=['GPT-4', 'GPT-3.5'],
                                    label='🧠 AI Model',
                                    value='GPT-4',
                                    info='GPT-4 daha yetenekli ama biraz daha yavaş'
                                )
                            with gr.Column():
                                headless = gr.Checkbox(
                                    label='🔲 Arka Planda Çalıştır',
                                    value=True,
                                    info='İşlem sırasında tarayıcıyı gizler'
                                )
                                max_steps = gr.Slider(
                                    minimum=5,
                                    maximum=20,
                                    value=10,
                                    step=1,
                                    label='🔄 Maksimum Adım Sayısı',
                                    info='Görevin tamamlanması için izin verilen maksimum adım sayısı'
                                )
                        
                        submit_btn = gr.Button('🚀 Görevi Başlat', variant='primary', scale=2)
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Sonuçlar")
                        output = gr.Textbox(
                            label="Terminal Output",
                            show_label=False,
                            lines=10
                        )
                        
            with gr.TabItem("📚 Örnekler"):
                example_tasks = [
                    ["Reddit'e git, 'browser-use' için arama yap, ilk gönderiye tıkla ve ilk yorumu getir"],
                    ["Amazon'da laptop ara, en iyi puanlılara göre sırala ve ilk 3 ürünün fiyatını listele"],
                    ["Google'da 'yapay zeka nedir' ara ve ilk 3 sonucun başlıklarını getir"],
                    ["YouTube'da en popüler videoları aç ve ilk 5 videonun başlık ve görüntülenme sayısını listele"]
                ]
                gr.Examples(
                    examples=example_tasks,
                    inputs=task,
                    label="Örnek Görevler",
                    examples_per_page=4
                )
            
            with gr.TabItem("ℹ️ Bilgi"):
                gr.Markdown("""
                ### 🤖 Web Otomasyon Asistanı Nasıl Çalışır?
                
                1. **Görev Tanımlama**: İstediğiniz web otomasyonu görevini doğal dilde yazın
                2. **Model Seçimi**: GPT-4 (daha yetenekli) veya GPT-3.5 (daha hızlı) modelini seçin
                3. **Çalışma Modu**: Tarayıcıyı görmek istiyorsanız "Arka Planda Çalıştır" seçeneğini kapatın
                4. **Adım Sayısı**: Karmaşık görevler için maksimum adım sayısını artırın
                
                ### 🎯 Örnek Kullanım Alanları
                
                - Web araştırması ve veri toplama
                - Fiyat karşılaştırma ve ürün analizi
                - Sosyal medya içerik analizi
                - Otomatik form doldurma ve test
                
                ### 💡 İpuçları
                
                - Görev açıklamanızı mümkün olduğunca net ve detaylı yazın
                - Karmaşık görevler için GPT-4 modelini tercih edin
                - Sonuçları kontrol etmek için tarayıcıyı görünür modda çalıştırın
                """)

        submit_btn.click(
            fn=lambda *args: asyncio.run(run_browser_task(*args)),
            inputs=[task, headless, model_choice, max_steps],
            outputs=output,
        )

    return interface

if __name__ == '__main__':
    print("Starting application...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    try:
        demo = create_ui()
        print("Launching Gradio interface...")
        demo.launch()
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        sys.exit(1)
