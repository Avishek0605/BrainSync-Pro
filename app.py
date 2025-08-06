# ============================================================================= 
# SECTION 1: IMPORTS AND CONFIGURATION
# =============================================================================

"""
✨ BrainSync Pro - Elite AI Intelligence Platform
===============================================
Section 1: Core imports and configuration setup
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Install required packages (run in Colab)
try:
    import gradio as gr
except ImportError:
    os.system('pip install gradio')
    import gradio as gr

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import RunnableSequence
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    os.system('pip install langchain langchain-google-genai')
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain_core.runnables import RunnableSequence
    from langchain.memory import ConversationBufferWindowMemory

@dataclass
class AppConfig:
    """Application configuration class"""
    API_KEY: str = "AIzaSyCTmtoeFy4Fvc83IIUobRqnEQlM-SosAzc"
    MODEL_NAME: str = "gemini-1.5-flash"
    TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 1000
    MEMORY_WINDOW: int = 10
    APP_TITLE: str = "✨ BrainSync Pro"
    APP_SUBTITLE: str = "Elite AI Intelligence Platform"
    APP_DESCRIPTION: str = "Next-generation AI assistant powered by advanced neural networks and cutting-edge language models"

# Premium color scheme and styling
THEME_COLORS = {
    'primary': '#6C5CE7',      # Premium Purple
    'secondary': '#FD79A8',    # Elegant Pink
    'accent': '#00D2D3',       # Cyan Accent
    'success': '#00B894',      # Modern Green
    'warning': '#FDCB6E',      # Gold Warning
    'danger': '#E84393',       # Stylish Red
    'dark': '#2D3436',         # Premium Dark
    'light': '#DDD6FE',        # Soft Light
    'gradient_1': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient_2': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    'gradient_3': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    'gradient_4': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)'
}

print("✅ Section 1: Imports and Configuration - LOADED!")

# =============================================================================
# SECTION 2: CORE AI SYSTEM CLASS
# =============================================================================

class ProfessionalQASystem:
    """
    Professional Question-Answering System using LangChain and Gemini AI
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = None
        self.chain = None
        self.memory = ConversationBufferWindowMemory(
            k=config.MEMORY_WINDOW,
            return_messages=True
        )
        self.conversation_history = []
        self.setup_model()
        self.create_chains()
    
    def setup_model(self) -> None:
        """Initialize the Gemini model with error handling"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.MODEL_NAME,
                google_api_key=self.config.API_KEY,
                temperature=self.config.TEMPERATURE,
                max_output_tokens=self.config.MAX_TOKENS
            )
            print("✅ Gemini model initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing model: {e}")
            raise
    
    def create_chains(self) -> None:
        """Create different types of prompt chains"""
        
        # Standard Q&A Chain
        self.qa_template = PromptTemplate(
            template="""You are a knowledgeable AI assistant. Provide clear, accurate, and helpful answers.

Question: {question}

Answer: Provide a comprehensive response that is informative and easy to understand.""",
            input_variables=["question"]
        )
        
        # Creative Writing Chain
        self.creative_template = PromptTemplate(
            template="""You are a creative writing assistant. Create engaging and imaginative content.

Request: {request}
Style: {style}

Creative Response:""",
            input_variables=["request", "style"]
        )
        
        # Professional Analysis Chain
        self.analysis_template = PromptTemplate(
            template="""You are a professional analyst. Provide detailed analysis and insights.

Topic: {topic}
Focus: {focus}

Professional Analysis:""",
            input_variables=["topic", "focus"]
        )
        
        # Create runnable chains
        self.qa_chain = self.qa_template | self.llm
        self.creative_chain = self.creative_template | self.llm
        self.analysis_chain = self.analysis_template | self.llm
        
        print("✅ Prompt chains created successfully!")
    
    def ask_question(self, question: str, chain_type: str = "qa") -> Tuple[str, bool]:
        """
        Process a question and return the answer
        """
        if not question.strip():
            return "Please enter a valid question.", False
        
        try:
            start_time = time.time()
            
            if chain_type == "qa":
                response = self.qa_chain.invoke({"question": question})
            elif chain_type == "creative":
                response = self.creative_chain.invoke({
                    "request": question, 
                    "style": "engaging and creative"
                })
            elif chain_type == "analysis":
                response = self.analysis_chain.invoke({
                    "topic": question,
                    "focus": "comprehensive analysis"
                })
            else:
                response = self.qa_chain.invoke({"question": question})
            
            processing_time = round(time.time() - start_time, 2)
            answer = response.content
            
            # Store in conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question,
                "answer": answer,
                "processing_time": processing_time,
                "chain_type": chain_type
            })
            
            return answer, True
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, False
    
    def get_conversation_summary(self) -> str:
        """Get a formatted summary of the conversation history"""
        if not self.conversation_history:
            return "No conversation history available."
        
        summary = f"📊 **Conversation Summary** ({len(self.conversation_history)} interactions)\n\n"
        
        for i, interaction in enumerate(self.conversation_history[-5:], 1):  # Show last 5
            summary += f"**{i}. [{interaction['timestamp']}]**\n"
            summary += f"❓ **Q:** {interaction['question'][:100]}{'...' if len(interaction['question']) > 100 else ''}\n"
            summary += f"✅ **A:** {interaction['answer'][:150]}{'...' if len(interaction['answer']) > 150 else ''}\n"
            summary += f"⏱️ *Processing time: {interaction['processing_time']}s*\n\n"
        
        return summary

print("✅ Section 2: Core AI System Class - LOADED!")

# =============================================================================
# SECTION 3: PREMIUM CSS STYLING
# =============================================================================

def get_premium_css():
    """Return premium CSS styling"""
    return """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        min-height: 100vh;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-header h1 {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #fff, #f0f8ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header .subtitle {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #E8F4FD !important;
        margin-bottom: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .main-header .description {
        font-size: 1.1rem !important;
        color: #B8D4E3 !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
    }
    
    .premium-card {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        margin: 1rem 0 !important;
    }
    
    .feature-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 18px;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(240, 147, 251, 0.3);
        border: none;
        backdrop-filter: blur(10px);
    }
    
    .feature-box h4 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .stats-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 2rem;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(67, 233, 123, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stats-box h3 {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    }
    
    .btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    }
    
    .gradio-textbox, .gradio-dropdown {
        border-radius: 12px !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .gradio-textbox:focus, .gradio-dropdown:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2) !important;
    }
    
    .footer {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin-top: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem !important;
        }
        .main-header .subtitle {
            font-size: 1.2rem !important;
        }
        .premium-card {
            padding: 1.5rem !important;
        }
    }
    """

print("✅ Section 3: Premium CSS Styling - LOADED!")
# =============================================================================
# SECTION 4: GRADIO INTERFACE (PART 1)
# =============================================================================

def create_professional_interface(qa_system: ProfessionalQASystem):
    """Create a professional Gradio interface"""
    
    custom_css = get_premium_css()
    
    with gr.Blocks(
        title="BrainSync Pro - Elite AI Platform",
        css=custom_css,
        theme=gr.themes.Glass()
    ) as demo:
        
        # Premium Header
        gr.HTML(f"""
        <div class="main-header">
            <h1>{qa_system.config.APP_TITLE}</h1>
            <div class="subtitle">{qa_system.config.APP_SUBTITLE}</div>
            <div class="description">{qa_system.config.APP_DESCRIPTION}</div>
            <div style="margin-top: 1.5rem; font-size: 0.95rem; opacity: 0.9;">
                🚀 Powered by Google Gemini AI • 🔗 Built with LangChain • ⚡ Lightning Fast
            </div>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # Main Q&A Tab
            with gr.Tab("🧠 Neural Q&A", elem_id="qa-tab"):
                with gr.Row():
                    with gr.Column(scale=2, elem_classes="premium-card"):
                        question_input = gr.Textbox(
                            label="💭 Ask Your Question",
                            placeholder="What would you like to know? (e.g., 'Explain quantum computing', 'What is machine learning?')",
                            lines=3,
                            max_lines=5,
                            elem_classes="premium-input"
                        )
                        
                        with gr.Row():
                            ask_btn = gr.Button("🚀 Generate Answer", variant="primary", size="lg", elem_classes="btn")
                            clear_btn = gr.Button("🗑️ Clear All", variant="secondary", elem_classes="btn")
                        
                        chain_type = gr.Radio(
                            choices=[
                                ("🎯 Smart Q&A", "qa"),
                                ("🎨 Creative Mode", "creative"), 
                                ("📊 Analysis Mode", "analysis")
                            ],
                            value="qa",
                            label="🔧 Intelligence Type",
                            info="Choose your preferred AI response style"
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>🌟 Elite Features</h4>
                            <ul style="line-height: 1.8; font-size: 0.95rem;">
                                <li>🧠 Advanced Neural Processing</li>
                                <li>🎨 Creative Intelligence Mode</li>
                                <li>📊 Professional Analysis</li>
                                <li>💾 Memory Retention</li>
                                <li>⚡ Ultra-Fast Response</li>
                                <li>🔒 Secure & Private</li>
                            </ul>
                        </div>
                        """)
                
                answer_output = gr.Textbox(
                    label="🤖 BrainSync Response",
                    lines=12,
                    max_lines=20,
                    show_copy_button=True,
                    elem_classes="premium-output"
                )
                
                status_output = gr.HTML()
        
        return demo, question_input, ask_btn, clear_btn, chain_type, answer_output, status_output

print("✅ Section 4: Gradio Interface (Part 1) - LOADED!")

# =============================================================================
# SECTION 5: INTERFACE PART 2 & EVENT HANDLERS
# =============================================================================

def setup_event_handlers(demo, qa_system, question_input, ask_btn, clear_btn, chain_type, answer_output, status_output):
    """Setup all event handlers for the interface"""
    
    def process_question(question, chain_type):
        if not question.strip():
            return "", "<div style='color: red;'>⚠️ Please enter a question</div>"
        
        answer, success = qa_system.ask_question(question, chain_type)
        
        if success:
            status = f"<div style='color: green;'>✅ Response generated successfully!</div>"
        else:
            status = f"<div style='color: red;'>❌ Error generating response</div>"
        
        return answer, status
    
    def clear_inputs():
        return "", ""
    
    # Connect events
    ask_btn.click(
        process_question,
        inputs=[question_input, chain_type],
        outputs=[answer_output, status_output]
    )
    
    clear_btn.click(
        clear_inputs,
        outputs=[question_input, answer_output]
    )
    
    return demo

def create_complete_interface(qa_system: ProfessionalQASystem):
    """Create the complete interface with all components"""
    
    custom_css = get_premium_css()
    
    with gr.Blocks(
        title="BrainSync Pro - Elite AI Platform",
        css=custom_css,
        theme=gr.themes.Glass()
    ) as demo:
        
        # Premium Header
        gr.HTML(f"""
        <div class="main-header">
            <h1>{qa_system.config.APP_TITLE}</h1>
            <div class="subtitle">{qa_system.config.APP_SUBTITLE}</div>
            <div class="description">{qa_system.config.APP_DESCRIPTION}</div>
            <div style="margin-top: 1.5rem; font-size: 0.95rem; opacity: 0.9;">
                🚀 Powered by Google Gemini AI • 🔗 Built with LangChain • ⚡ Lightning Fast
            </div>
        </div>
        """)
        
        with gr.Tabs() as tabs:
            
            # Main Q&A Tab
            with gr.Tab("🧠 Neural Q&A", elem_id="qa-tab"):
                with gr.Row():
                    with gr.Column(scale=2, elem_classes="premium-card"):
                        question_input = gr.Textbox(
                            label="💭 Ask Your Question",
                            placeholder="What would you like to know? (e.g., 'Explain quantum computing', 'What is machine learning?')",
                            lines=3,
                            max_lines=5
                        )
                        
                        with gr.Row():
                            ask_btn = gr.Button("🚀 Generate Answer", variant="primary", size="lg")
                            clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                        
                        chain_type = gr.Radio(
                            choices=[
                                ("🎯 Smart Q&A", "qa"),
                                ("🎨 Creative Mode", "creative"), 
                                ("📊 Analysis Mode", "analysis")
                            ],
                            value="qa",
                            label="🔧 Intelligence Type"
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="feature-box">
                            <h4>🌟 Elite Features</h4>
                            <ul style="line-height: 1.8; font-size: 0.95rem;">
                                <li>🧠 Advanced Neural Processing</li>
                                <li>🎨 Creative Intelligence Mode</li>
                                <li>📊 Professional Analysis</li>
                                <li>💾 Memory Retention</li>
                                <li>⚡ Ultra-Fast Response</li>
                                <li>🔒 Secure & Private</li>
                            </ul>
                        </div>
                        """)
                
                answer_output = gr.Textbox(
                    label="🤖 BrainSync Response",
                    lines=12,
                    max_lines=20,
                    show_copy_button=True
                )
                
                status_output = gr.HTML()
        
        # Event handlers
        def process_question(question, chain_type_val):
            if not question.strip():
                return "", "<div style='color: red;'>⚠️ Please enter a question</div>"
            
            answer, success = qa_system.ask_question(question, chain_type_val)
            
            if success:
                status = f"<div style='color: green;'>✅ Response generated successfully!</div>"
            else:
                status = f"<div style='color: red;'>❌ Error generating response</div>"
            
            return answer, status
        
        def clear_inputs():
            return "", ""
        
        # Connect events
        ask_btn.click(
            process_question,
            inputs=[question_input, chain_type],
            outputs=[answer_output, status_output]
        )
        
        clear_btn.click(
            clear_inputs,
            outputs=[question_input, answer_output]
        )
        
        # Premium Footer
        gr.HTML("""
        <div class="footer">
            <h3 style="margin-bottom: 1rem; font-size: 1.5rem;">✨ BrainSync Pro</h3>
            <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">🚀 Built with LangChain & Gradio | Powered by Google Gemini AI</p>
            <p style="font-size: 1rem; opacity: 0.9;"><em>🎯 Elite AI Assistant for Next-Generation Productivity</em></p>
            <div style="margin-top: 1.5rem; font-size: 0.9rem; opacity: 0.8;">
                🌟 Premium Experience • 🔒 Secure Processing • ⚡ Lightning Performance
            </div>
        </div>
        """)
    
    return demo

print("✅ Section 5: Interface Part 2 & Event Handlers - LOADED!")

# =============================================================================
# SECTION 6: MAIN APPLICATION & LAUNCH
# =============================================================================

def main():
    """Main application entry point"""
    print("🚀 Initializing BrainSync Pro...")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config = AppConfig()
        
        # Create QA system
        qa_system = ProfessionalQASystem(config)
        
        # Test basic functionality
        print("\n🧪 Testing system functionality...")
        test_answer, test_success = qa_system.ask_question("What is artificial intelligence?")
        
        if test_success:
            print("✅ System test passed!")
            print(f"📝 Test response: {test_answer[:100]}...")
        else:
            print("❌ System test failed!")
            return
        
        # Create and launch interface
        print("\n🎨 Creating professional interface...")
        demo_interface = create_complete_interface(qa_system)
        
        print("\n🌟 BrainSync Pro Ready!")
        print("📱 Launching web interface...")
        print("=" * 60)
        
        # Launch with professional settings
        demo_interface.launch(
            share=True,
            debug=False,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Please check your API key and internet connection.")

# CLI mode for command line usage
def cli_mode():
    """Command-line interface mode"""
    print("\n" + "="*60)
    print("🖥️  COMMAND LINE MODE")
    print("="*60)
    
    config = AppConfig()
    qa_system = ProfessionalQASystem(config)
    
    print("✅ System ready! Type 'quit' to exit, 'history' to see conversation history.")
    
    while True:
        try:
            question = input("\n💭 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'history':
                print(qa_system.get_conversation_summary())
                continue
            elif not question:
                print("⚠️  Please enter a question.")
                continue
            
            print("🤔 Thinking...")
            answer, success = qa_system.ask_question(question)
            
            if success:
                print(f"\n🤖 AI Response:\n{answer}")
            else:
                print(f"\n❌ Error: {answer}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

# Execution control
if __name__ == "__main__":
    # Check if running in Colab or Jupyter
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    # Display startup
    
    # =============================================================================
# SECTION 7: FINAL LAUNCH & EXECUTION
# =============================================================================

# Complete the main function from Section 6
def complete_startup():
    """Complete startup information and launch"""
    
    # Check if running in Colab or Jupyter
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    # Display startup information
    print("🚀 BrainSync Pro - Elite AI Intelligence Platform")
    print("=" * 60)
    print(f"📱 Environment: {'Google Colab' if IN_COLAB else 'Local/Jupyter'}")
    print(f"🤖 Model: Gemini 1.5 Flash")
    print(f"🔗 Framework: LangChain")
    print("=" * 60)
    
    # Launch the application
    main()

# Simple test function to verify everything works
def quick_test():
    """Quick test to verify the system works"""
    print("🧪 Running Quick Test...")
    
    try:
        config = AppConfig()
        qa_system = ProfessionalQASystem(config)
        
        # Test a simple question
        answer, success = qa_system.ask_question("Hello, how are you?")
        
        if success:
            print("✅ Quick test PASSED!")
            print(f"🤖 Response: {answer[:100]}...")
            return True
        else:
            print("❌ Quick test FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

# Launch function for Colab
def launch_brainsync_pro():
    """Main launch function for BrainSync Pro"""
    print("🎉 Starting BrainSync Pro...")
    
    # Run quick test first
    if quick_test():
        print("\n🚀 Launching full application...")
        complete_startup()
    else:
        print("❌ System test failed. Please check your setup.")

# Auto-run if this is the main execution
print("✅ Section 7: Final Launch & Execution - LOADED!")
print("\n🎯 ALL SECTIONS LOADED SUCCESSFULLY!")
print("=" * 60)
print("🚀 Ready to launch! Run: launch_brainsync_pro()")
print("=" * 60)

# Uncomment the line below to auto-launch
# launch_brainsync_pro()

launch_brainsync_pro()
