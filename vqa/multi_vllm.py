from openai import OpenAI
import os
import base64
import json
from enum import Enum
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List
import datetime
import shutil
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("mdt_consultation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 配置LLM模型设置
LLM_MODELS_SETTINGS = {
    "deepseek-v3-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat",
        "comment": "DeepSeek V3 官方站点",
        "reasoning": False,
    },
    "deepseek-r1-official": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-reasoner",
        "comment": "DeepSeek R1 推理模型 官方站点",
        "reasoning": True,
    },
    "deepseek-v3-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-v3",
        "comment": "DeepSeek V3 阿里站点",
        "reasoning": False,
    },
    "deepseek-r1-ali": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "deepseek-r1",
        "comment": "DeepSeek R1 推理模型 阿里站点",
        "reasoning": True,
    },
    "qwen-max-latest": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-max-latest",
        "comment": "通义千问 Max",
        "reasoning": False,
    },
    "qwen-vl-max": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-vl-max",
        "comment": "qwen-vl-max",
        "reasoning": False,
    }
}


def encode_image(image_path: str) -> str:
    """
    将图像文件编码为base64字符串。
    
    参数:
        image_path: 图像文件路径
        
    返回:
        图像的Base64编码字符串
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"图像文件未找到: {image_path}")
    except IOError as e:
        raise IOError(f"读取图像文件错误: {e}")


class MedicalSpecialty(Enum):
    """医学专科枚举类。"""
    INTERNAL_MEDICINE = "内科"
    SURGERY = "外科"
    RADIOLOGY = "放射科"


class AgentType(Enum):
    """智能体类型枚举类。"""
    DOCTOR = "医生"
    META = "协调者"


class BaseAgent:
    """所有智能体的基类。"""
    
    def __init__(self, 
                 agent_id: str, 
                 agent_type: AgentType, 
                 model_key: str = "qwen-vl-max"):
        """
        初始化基础智能体。
        
        参数:
            agent_id: 智能体的唯一标识符
            agent_type: 智能体类型(医生或协调者)
            model_key: 使用的LLM模型
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.model_key = model_key
        
        if model_key not in LLM_MODELS_SETTINGS:
            raise ValueError(f"模型键'{model_key}'在LLM_MODELS_SETTINGS中未找到")
        
        # 根据模型配置设置OpenAI客户端
        model_settings = LLM_MODELS_SETTINGS[model_key]
        self.client = OpenAI(
            api_key=model_settings["api_key"],
            base_url=model_settings["base_url"],
        )
        self.model_name = model_settings["model_name"]
    
    def call_llm(self, 
                system_message: Dict[str, str], 
                user_message: Dict[str, Any]) -> str:
        """
        调用LLM，提供消息。
        
        参数:
            system_message: 系统消息，设置上下文
            user_message: 用户消息，包含提示和图像
            
        返回:
            LLM的响应文本
        """
        try:
            logger.debug(f"智能体 {self.agent_id} 调用LLM，系统消息: {system_message['content'][:50]}...")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[system_message, user_message],
                response_format={"type": "json_object"}
            )
            response = completion.choices[0].message.content
            logger.debug(f"智能体 {self.agent_id} 收到响应: {response[:50]}...")
            return response
        except Exception as e:
            logger.error(f"调用LLM API错误: {e}")
            raise Exception(f"调用LLM API错误: {e}")


class DoctorAgent(BaseAgent):
    """具有医学专科的医生智能体。"""
    
    def __init__(self, 
                 agent_id: str, 
                 specialty: MedicalSpecialty,
                 model_key: str = "qwen-vl-max"):
        """
        初始化医生智能体。
        
        参数:
            agent_id: 智能体的唯一标识符
            specialty: 医生的医学专科
            model_key: 使用的LLM模型
        """
        super().__init__(agent_id, AgentType.DOCTOR, model_key)
        self.specialty = specialty
        # 新增：维护医生的决策历史记忆
        self.memory = []
        logger.info(f"初始化{specialty.value}医生智能体，ID: {agent_id}")
    
    def analyze_case(self, 
                    image_path: str, 
                    prompt: str,
                    options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        分析医疗病例。
        
        参数:
            image_path: 医学图像路径
            prompt: 关于病例的问题
            options: 可选的多选项选项
            
        返回:
            包含分析结果的字典
        """
        logger.info(f"医生 {self.agent_id} ({self.specialty.value}) 正在分析病例")
        base64_image = encode_image(image_path)
        
        # 准备系统消息以指导医生分析
        system_message = {
            "role": "system",
            "content": f"你是一名专攻{self.specialty.value}的医生。"
                      f"请分析此医学图像并提供你对问题的专业意见。"
                      f"你的输出应为JSON格式，包含'explanation'(详细推理)和"
                      f"'answer'(明确结论)字段。"
        }
        
        # 对于多选题，指示选择一个选项
        if options:
            system_message["content"] += (
                f"对于多选题，请确保你的'answer'字段包含所选选项的完整文本，而不仅仅是选项编号。"
            )
        
        # 准备用户消息内容
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ]
        
        # 如果提供选项，则添加到提示中
        if options:
            options_text = "\n选项:\n" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            prompt_with_options = f"{prompt}\n{options_text}"
        else:
            prompt_with_options = prompt
        
        user_content.append({
            "type": "text", 
            "text": f"{prompt_with_options}\n\n请提供你的分析，以JSON格式返回，包含'explanation'和'answer'字段。"
        })
        
        user_message = {
            "role": "user",
            "content": user_content,
        }
        
        # 调用LLM
        response_text = self.call_llm(system_message, user_message)
        
        # 解析响应
        try:
            result = json.loads(response_text)
            logger.info(f"医生 {self.agent_id} 响应成功解析")
            # 添加到记忆中
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # 如果JSON格式不正确，使用备用解析
            logger.warning(f"医生 {self.agent_id} 响应不是有效JSON，使用备用解析")
            result = parse_structured_output(response_text)
            # 添加到记忆中
            self.memory.append({
                "type": "analysis",
                "round": len(self.memory) // 2 + 1,
                "content": result
            })
            return result
    
    def review_synthesis(self, 
                         image_path: str,
                         prompt: str, 
                         synthesis: Dict[str, Any],
                         options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        审查元智能体的综合结果。
        
        参数:
            image_path: 医学图像路径
            prompt: 原始问题
            synthesis: 元智能体的综合结果
            options: 可选的多选项选项
            
        返回:
            包含同意状态和可能的反驳理由的字典
        """
        logger.info(f"医生 {self.agent_id} ({self.specialty.value}) 正在审查综合结果")
        base64_image = encode_image(image_path)
        
        # 获取当前轮次
        current_round = len(self.memory) // 2 + 1
        
        # 获取医生自己的最近分析
        own_analysis = None
        for mem in reversed(self.memory):
            if mem["type"] == "analysis":
                own_analysis = mem["content"]
                break
        
        # 准备审查的系统消息
        system_message = {
            "role": "system",
            "content": f"你是一名专攻{self.specialty.value}的医生，正在参与第{current_round}轮多学科团队会诊。"
                      f"请审查多个医生意见的综合结果，并确定你是否同意该结论。"
                      f"考虑你之前的分析和MetaAgent的综合意见，决定是否同意或提出不同意见。"
                      f"你的输出应为JSON格式，包含'agree'(布尔值或'yes'/'no')、'reason'(你的决定理由)，"
                      f"以及'answer'(如果不同意，提供你的建议答案；如果同意，可以重复综合答案)字段。"
        }
        
        # 准备包含综合结果的用户消息
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ]
        
        # 如果提供选项，则添加到提示中
        if options:
            options_text = "\n选项:\n" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            prompt_with_options = f"{prompt}\n{options_text}"
        else:
            prompt_with_options = prompt
        
        # 准备包含自己之前分析的文本
        own_analysis_text = ""
        if own_analysis:
            own_analysis_text = f"你的先前分析:\n解释: {own_analysis.get('explanation', '')}\n答案: {own_analysis.get('answer', '')}\n\n"
        
        synthesis_text = f"综合解释: {synthesis.get('explanation', '')}\n"
        synthesis_text += f"建议答案: {synthesis.get('answer', '')}"
        
        user_content.append({
            "type": "text", 
            "text": f"原始问题: {prompt_with_options}\n\n"
                  f"{own_analysis_text}"
                  f"{synthesis_text}\n\n"
                  f"你是否同意这个综合结果？请以JSON格式提供回答，包含:\n"
                  f"1. 'agree': 'yes'/'no'\n"
                  f"2. 'reason': 你同意或不同意的理由\n"
                  f"3. 'answer': 你支持的答案（如果同意可以是综合答案，如果不同意需要提供你的建议答案）"
        })
        
        user_message = {
            "role": "user",
            "content": user_content,
        }
        
        # 调用LLM
        response_text = self.call_llm(system_message, user_message)
        
        # 解析响应
        try:
            result = json.loads(response_text)
            logger.info(f"医生 {self.agent_id} 审查成功解析")
            
            # 标准化agree字段
            if isinstance(result.get("agree"), str):
                result["agree"] = result["agree"].lower() in ["true", "yes", "是", "同意"]
            
            # 添加到记忆中
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result
        except json.JSONDecodeError:
            # 备用解析
            logger.warning(f"医生 {self.agent_id} 审查不是有效JSON，使用备用解析")
            lines = response_text.strip().split('\n')
            result = {}
            
            for line in lines:
                if "agree" in line.lower():
                    result["agree"] = "true" in line.lower() or "yes" in line.lower() or "是" in line.lower() or "同意" in line.lower()
                elif "reason" in line.lower():
                    result["reason"] = line.split(":", 1)[1].strip() if ":" in line else line
                elif "answer" in line.lower():
                    result["answer"] = line.split(":", 1)[1].strip() if ":" in line else line
            
            # 确保必填字段
            if "agree" not in result:
                result["agree"] = False
            if "reason" not in result:
                result["reason"] = "未提供理由"
            if "answer" not in result:
                # 默认使用自己之前的答案或者综合答案
                if own_analysis and "answer" in own_analysis:
                    result["answer"] = own_analysis["answer"]
                else:
                    result["answer"] = synthesis.get("answer", "未提供答案")
            
            # 添加到记忆中
            self.memory.append({
                "type": "review",
                "round": current_round,
                "content": result
            })
            return result


class MetaAgent(BaseAgent):
    """元智能体，综合多个医生意见。"""
    
    def __init__(self, agent_id: str, model_key: str = "qwen-vl-max"):
        """
        初始化元智能体。
        
        参数:
            agent_id: 智能体的唯一标识符
            model_key: 使用的LLM模型
        """
        super().__init__(agent_id, AgentType.META, model_key)
        logger.info(f"初始化元智能体，ID: {agent_id}")
    
    def synthesize_opinions(self, 
                           image_path: str, 
                           prompt: str,
                           doctor_opinions: List[Dict[str, Any]],
                           doctor_specialties: List[MedicalSpecialty],
                           current_round: int = 1,
                           options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        综合多个医生意见。
        
        参数:
            image_path: 医学图像路径
            prompt: 原始问题
            doctor_opinions: 医生意见列表
            doctor_specialties: 对应的医生专科列表
            current_round: 当前讨论轮次
            options: 可选的多选项选项
            
        返回:
            包含综合解释和答案的字典
        """
        logger.info(f"元智能体正在综合第{current_round}轮意见")
        base64_image = encode_image(image_path)
        
        # 准备用于综合的系统消息
        system_message = {
            "role": "system",
            "content": f"你是一名医疗共识协调者，正在主持第{current_round}轮多学科团队会诊。"
                      "请将多个专科医生的意见综合成一个连贯的分析和结论。"
                      "考虑每位医生的专业知识和观点，并相应地权衡他们的意见。"
                      "你的输出应为JSON格式，包含'explanation'(推理综合)和"
                      "'answer'(共识结论)字段。"
        }
        
        # 对于多选题，指示选择一个选项
        if options:
            system_message["content"] += (
                f"对于多选题，请确保你的'answer'字段包含所选选项的完整文本，而不仅仅是选项编号。"
            )
        
        # 格式化医生意见作为输入
        formatted_opinions = []
        for i, (opinion, specialty) in enumerate(zip(doctor_opinions, doctor_specialties)):
            formatted_opinion = f"医生 {i+1} ({specialty.value}):\n"
            formatted_opinion += f"解释: {opinion.get('explanation', '')}\n"
            formatted_opinion += f"答案: {opinion.get('answer', '')}\n"
            formatted_opinions.append(formatted_opinion)
        
        opinions_text = "\n".join(formatted_opinions)
        
        # 准备包含所有意见的用户消息
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ]
        
        # 如果提供选项，则添加到提示中
        if options:
            options_text = "\n选项:\n" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            prompt_with_options = f"{prompt}\n{options_text}"
        else:
            prompt_with_options = prompt
        
        user_content.append({
            "type": "text", 
            "text": f"问题: {prompt_with_options}\n\n"
                  f"第{current_round}轮医生意见:\n{opinions_text}\n\n"
                  f"请将这些意见综合成一个共识观点。以JSON格式提供你的综合，包含"
                  f"'explanation'(全面推理)和'answer'(明确结论)字段。"
        })
        
        user_message = {
            "role": "user",
            "content": user_content,
        }
        
        # 调用LLM
        response_text = self.call_llm(system_message, user_message)
        
        # 解析响应
        try:
            result = json.loads(response_text)
            logger.info("元智能体综合成功解析")
            return result
        except json.JSONDecodeError:
            # 备用解析
            logger.warning("元智能体综合不是有效JSON，使用备用解析")
            return parse_structured_output(response_text)
    
    def make_final_decision(self, 
                           image_path: str,
                           prompt: str,
                           doctor_reviews: List[Dict[str, Any]],
                           doctor_specialties: List[MedicalSpecialty],
                           current_synthesis: Dict[str, Any],
                           current_round: int,
                           max_rounds: int,
                           options: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        基于医生审查做出最终决定。
        
        参数:
            image_path: 医学图像路径
            prompt: 原始问题
            doctor_reviews: 医生审查列表
            doctor_specialties: 对应的医生专科列表
            current_synthesis: 当前综合结果
            current_round: 当前轮次
            max_rounds: 最大轮次
            options: 可选的多选项选项
            
        返回:
            包含最终解释和答案的字典
        """
        logger.info(f"元智能体正在做出第{current_round}轮决定")
        base64_image = encode_image(image_path)
        
        # 检查所有医生是否都同意
        all_agree = all(review.get('agree', False) for review in doctor_reviews)
        reached_max_rounds = current_round >= max_rounds
        
        # 准备最终决定的系统消息
        system_message = {
            "role": "system",
            "content": "你是一名做出最终决定的医疗共识协调者。"
        }
        
        if all_agree:
            system_message["content"] += "所有医生都同意你的综合结果，请生成最终报告。"
        elif reached_max_rounds:
            system_message["content"] += (
                f"已达到最大讨论轮数({max_rounds}轮)，但仍未达成完全共识。"
                f"请使用多数意见方法做出最终决定。"
            )
        else:
            system_message["content"] += (
                "并非所有医生都同意你的综合结果，但需要给出当前轮次的决定。"
            )
        
        system_message["content"] += (
            "你的输出应为JSON格式，包含'explanation'(最终推理)和"
            "'answer'(最终结论)字段。"
        )
        
        # 对于多选题，指示选择一个选项
        if options:
            system_message["content"] += (
                f"对于多选题，请确保你的'answer'字段包含所选选项的完整文本，而不仅仅是选项编号。"
            )
        
        # 格式化医生审查
        formatted_reviews = []
        for i, (review, specialty) in enumerate(zip(doctor_reviews, doctor_specialties)):
            formatted_review = f"医生 {i+1} ({specialty.value}):\n"
            formatted_review += f"同意: {'是' if review.get('agree', False) else '否'}\n"
            formatted_review += f"理由: {review.get('reason', '')}\n"
            formatted_review += f"答案: {review.get('answer', '')}\n"
            formatted_reviews.append(formatted_review)
        
        reviews_text = "\n".join(formatted_reviews)
        
        # 准备用户消息
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            }
        ]
        
        # 如果提供选项，则添加到提示中
        if options:
            options_text = "\n选项:\n" + "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
            prompt_with_options = f"{prompt}\n{options_text}"
        else:
            prompt_with_options = prompt
        
        current_synthesis_text = (
            f"当前综合解释: {current_synthesis.get('explanation', '')}\n"
            f"当前建议答案: {current_synthesis.get('answer', '')}"
        )
        
        decision_type = "最终" if all_agree or reached_max_rounds else "当前轮次"
        
        user_content.append({
            "type": "text", 
            "text": f"问题: {prompt_with_options}\n\n"
                  f"{current_synthesis_text}\n\n"
                  f"医生审查:\n{reviews_text}\n\n"
                  f"请提供你的{decision_type}决定，"
                  f"以JSON格式，包含'explanation'和'answer'字段。"
        })
        
        user_message = {
            "role": "user",
            "content": user_content,
        }
        
        # 调用LLM
        response_text = self.call_llm(system_message, user_message)
        
        # 解析响应
        try:
            result = json.loads(response_text)
            logger.info("元智能体最终决定成功解析")
            return result
        except json.JSONDecodeError:
            # 备用解析
            logger.warning("元智能体最终决定不是有效JSON，使用备用解析")
            return parse_structured_output(response_text)


class MDTConsultation:
    """多学科团队会诊协调器。"""
    
    def __init__(self, 
                max_rounds: int = 3,
                model_key: str = "qwen-vl-max",
                output_dir: str = "mdt_consultations"):
        """
        初始化MDT会诊。
        
        参数:
            max_rounds: 最大讨论回合数
            model_key: 所有智能体使用的LLM模型
            output_dir: 保存会诊结果的目录
        """
        self.max_rounds = max_rounds
        self.model_key = model_key
        self.output_dir = output_dir
        
        # 如果输出目录不存在，则创建
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化不同专科的医生智能体
        self.doctor_agents = [
            DoctorAgent("doctor_1", MedicalSpecialty.INTERNAL_MEDICINE, model_key),
            DoctorAgent("doctor_2", MedicalSpecialty.SURGERY, model_key),
            DoctorAgent("doctor_3", MedicalSpecialty.RADIOLOGY, model_key)
        ]
        
        # 初始化元智能体
        self.meta_agent = MetaAgent("meta", model_key)
        
        # 存储医生专科以便易于访问
        self.doctor_specialties = [doctor.specialty for doctor in self.doctor_agents]
        
        # 会诊历史
        self.consultation_history = []
        
        logger.info(f"初始化MDT会诊，max_rounds={max_rounds}, model_key={model_key}")
    
    def run_consultation(self, 
                        image_path: str, 
                        prompt: str,
                        options: Optional[List[str]] = None,
                        case_id: Optional[str] = None) -> Dict[str, Any]:
        """
        运行MDT会诊流程。
        
        参数:
            image_path: 医学图像路径
            prompt: 关于病例的问题
            options: 可选的多选项选项
            case_id: 可选的病例标识符
            
        返回:
            包含最终会诊结果的字典
        """
        # 如果未提供病例ID，则生成
        if case_id is None:
            case_id = f"case_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建病例目录
        case_dir = os.path.join(self.output_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        
        # 将图像复制到病例目录
        image_filename = os.path.basename(image_path)
        case_image_path = os.path.join(case_dir, image_filename)
        shutil.copy2(image_path, case_image_path)
        
        logger.info(f"开始病例 {case_id} 的MDT会诊")
        logger.info(f"问题: {prompt}")
        if options:
            logger.info(f"选项: {options}")
        
        # 本病例的会诊历史
        case_history = {
            "case_id": case_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "case": {
                "image_path": image_path,
                "case_image_path": case_image_path,
                "prompt": prompt,
                "options": options
            },
            "rounds": []
        }
        self.consultation_history.append(case_history)
        
        current_round = 0
        final_decision = None
        consensus_reached = False
        
        while current_round < self.max_rounds and not consensus_reached:
            current_round += 1
            logger.info(f"开始第 {current_round} 轮")
            
            round_data = {"round": current_round, "opinions": [], "synthesis": None, "reviews": []}
            
            # 步骤1: 每位医生分析病例
            doctor_opinions = []
            for i, doctor in enumerate(self.doctor_agents):
                logger.info(f"医生 {i+1} ({doctor.specialty.value}) 正在分析病例")
                opinion = doctor.analyze_case(image_path, prompt, options)
                doctor_opinions.append(opinion)
                round_data["opinions"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "opinion": opinion
                })
                
                logger.info(f"医生 {i+1} 意见: {opinion.get('answer', '')}")
            
            # 步骤2: 元智能体综合意见
            logger.info("元智能体正在综合意见")
            synthesis = self.meta_agent.synthesize_opinions(
                image_path, prompt, doctor_opinions, self.doctor_specialties, 
                current_round, options
            )
            round_data["synthesis"] = synthesis
            
            logger.info(f"元智能体综合: {synthesis.get('answer', '')}")
            
            # 步骤3: 医生审查综合结果
            doctor_reviews = []
            all_agree = True
            for i, doctor in enumerate(self.doctor_agents):
                logger.info(f"医生 {i+1} ({doctor.specialty.value}) 正在审查综合结果")
                review = doctor.review_synthesis(image_path, prompt, synthesis, options)
                doctor_reviews.append(review)
                round_data["reviews"].append({
                    "doctor_id": doctor.agent_id,
                    "specialty": doctor.specialty.value,
                    "review": review
                })
                
                agrees = review.get('agree', False)
                all_agree = all_agree and agrees
                
                logger.info(f"医生 {i+1} 同意: {'是' if agrees else '否'}")
            
            # 将回合数据添加到历史记录
            case_history["rounds"].append(round_data)
            
            # 步骤4: 元智能体根据审查做出决定
            decision = self.meta_agent.make_final_decision(
                image_path, prompt, doctor_reviews, self.doctor_specialties, 
                synthesis, current_round, self.max_rounds, options
            )
            
            # 检查是否达成共识
            if all_agree:
                consensus_reached = True
                final_decision = decision
                logger.info("达成共识")
            else:
                logger.info("未达成共识，继续下一轮")
                if current_round == self.max_rounds:
                    # 如果已达到最大轮次，使用最后一轮的决定作为最终决定
                    final_decision = decision
        
        # 将最终决定添加到历史记录
        if not final_decision:
            # 如果没有达到共识但用完了轮次，使用最后一轮的决定
            final_decision = decision
        
        case_history["final_decision"] = final_decision
        case_history["consensus_reached"] = consensus_reached
        case_history["total_rounds"] = current_round
        
        logger.info(f"最终决定: {final_decision.get('answer', '')}")
        
        # 将病例历史保存到文件
        case_history_path = os.path.join(case_dir, "consultation_history.json")
        with open(case_history_path, "w", encoding="utf-8") as f:
            json.dump(case_history, f, ensure_ascii=False, indent=2)
        
        # 保存完整会诊历史
        full_history_path = os.path.join(self.output_dir, "full_consultation_history.json")
        with open(full_history_path, "w", encoding="utf-8") as f:
            json.dump(self.consultation_history, f, ensure_ascii=False, indent=2)
        
        return final_decision


def parse_structured_output(response_text: str) -> Dict[str, str]:
    """
    解析LLM响应以提取结构化输出。
    
    参数:
        response_text: 来自LLM的文本响应
        
    返回:
        包含结构化字段的字典
    """
    try:
        # 尝试解析为JSON
        parsed = json.loads(response_text)
        return parsed
    except json.JSONDecodeError:
        # 如果不是有效JSON，从文本中提取
        # 这是模型未正确格式化JSON时的备用方案
        lines = response_text.strip().split('\n')
        result = {}
        
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                result[key] = value
        
        # 确保至少存在explanation和answer字段
        if "explanation" not in result:
            result["explanation"] = "响应中未找到结构化解释"
        if "answer" not in result:
            result["answer"] = "响应中未找到结构化答案"
            
        return result


def main():
    """演示MDT会诊的主函数。"""
    image_path = "vqa/my_datasets/test_img.jpg"
    prompt = "这张图像中显示的主要异常是什么？"
    options = [
        "肺结节",
        "胸腔积液",
        "气胸",
        "肺水肿"
    ]
    
    try:
        # 运行会诊
        mdt = MDTConsultation(max_rounds=3, model_key="qwen-vl-max")
        result = mdt.run_consultation(image_path, prompt, options)
        
        print("\n会诊完成!")
        print("最终答案:", result["answer"])
        print("解释:", result["explanation"])
        
        # 显示会诊历史保存位置
        print(f"会诊历史保存在 {mdt.output_dir}")
        
    except Exception as e:
        logger.error(f"MDT会诊错误: {e}")
        print(f"MDT会诊错误: {e}")


if __name__ == "__main__":
    main()