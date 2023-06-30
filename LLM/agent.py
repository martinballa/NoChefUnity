from typing import List, Union
import re
import os 
import time

from text_generation import Client
from .llm import HuggingFaceTextGenInference
import langchain
from langchain.memory import ConversationBufferWindowMemory

from langchain.agents import (
    Tool, 
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from .action_tool import ActionTool

from .agent_executor import ReACTAgentExecutor
from langchain.prompts import (
    StringPromptTemplate,
)
from langchain import (
    LLMChain,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


def generate_react_agent(
    llm_dict,
    inference_server_url = "https://api-inference.huggingface.co/models/timdettmers/guanaco-33b-merged",
    max_new_tokens = 1024,
    top_k = 10,
    top_p = 0.95,
    typical_p = 0.95,
    temperature = 0.1,
    repetition_penalty = 1.2,
    use_tool_retriever=True,
) -> WrappedAgent :

    os.environ["HF_API_TOKEN"] = "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"
    HF_TOKEN = os.environ.get("HF_API_TOKEN", "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")
    MODEL_NAME = "timdettmers/guanaco-33b-merged"
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

    llm = HuggingFaceTextGenInference(
        inference_server_url = inference_server_url,
        max_new_tokens = max_new_tokens,
        top_k = top_k,
        top_p = top_p,
        typical_p = typical_p,
        temperature = temperature,
        repetition_penalty = repetition_penalty,
    )
    llm_dict['llm'] = llm

    # Define tools :
    # These ActionTool's will make the whole ReAct loop hang
    # until the model provides a relevant observation for the 
    # loop to carry on.
    model_comm_filepath = "./model_com.txt" 
    moveto_tool = ActionTool(
        name="MoveTo",
        description="""
        Move the avatar to a specific position in the environment.
        The input must be a 2D tuple.
        """,
        model_comm_filepath=model_comm_filepath,
    )
    
    donothing_tool = ActionTool(
        name="DoNothing",
        description="""
        Sarcastically make the avatar do nothing.
        The input must be left blank.
        """,
        model_comm_filepath=model_comm_filepath,
    )
    
    react_tools = [
        moveto_tool,
        donothing_tool,
    ]
    
    # Profile :
    angry_profile = """
You are angry and this anger is slightly toxic to the room you are in.
"""
    
    # Prompt :
    react_template = """
React to the prompt while staying in character:
{profile}
You have access to the following actions:

{tools}

Use the following format:

Prompt: the input prompt that you must react to
Thought: you should always think about what to do
Action: the action to take in response to the prompt, should be one of [{tool_names}]
Action Input: the input to the action, if any. If none, then leave it blank.
Reply: the verbal reply that you want to give to the prompt, should be in between double quotation marks.
Observation: the result of the action.
... (this Thought/Action/Action Input/Reply/Observation/Prompt will repeat indefinitely)

Begin!

Prompt: {input}
{agent_scratchpad}
"""

    react_template_with_context = """
Context to consider:
{context}
"""
    react_template_with_context += "\n" + react_template
    
    SYSTEM_TAG = "#SYSTEM" #"<|SYSTEM|>""
    USER_TAG = "#USER" #"<|USER|>"
    AGENT_TAG = "#RAMSAY"
    class ReACTPromptTemplate(StringPromptTemplate):
        llm_dict: Dict[str, Any]
        # the template to use :
        template: str
        # the List of tools available:
        tools: List[Tool]
        tool_retriever: Any = None
        # the number of intermediate steps to include in the prompt:
        cot_k: int = 3
        # the profile to roleplay:
        profile: str

        def format(self, **kwargs) -> str:
            """
            Get the intermediate steps as (AgentAction, Obvservation)
            tuples and format them in a particular way.
            """
            time.sleep(2)
            intermediate_steps = kwargs.pop("intermediate_steps")
            thoughts = ""
            for action, observation in intermediate_steps[-self.cot_k:]:
                thoughts += action.log
                thoughts += f"\nObservation: {observation}\nThought: "
            # Scratchpad :
            kwargs["agent_scratchpad"] = thoughts
            
            # Tools :
            """
            if use_tool_retriever:
                if self.tool_retriever is None:
                    self.tool_retriever = generate_tool_retriever(
                        ALL_TOOLS=self.tools,
                        Embeddings=None,
                    )
                tools = get_tools(
                    query=kwargs['input'], 
                    retriever=self.tool_retriever,
                    ALL_TOOLS=self.tools,
                    k=2,
                )
            else:
                tools = self.tools
            """
            tools = self.tools

            kwargs["tools"] = "\n".join([
                f"{tool.name}:{tool.description}"
                for tool in tools
            ])
            kwargs["tool_names"] = ", ".join([
                tool.name
                for tool in tools
            ])
            
            kwargs['profile'] = self.profile

            # TODO : context should be provided at the higher level...
            kwargs['context'] = ''

            kwargs["SYSTEM_TAG"] = SYSTEM_TAG
            kwargs["USER_TAG"] = USER_TAG
            kwargs["AGENT_TAG"] = AGENT_TAG
            
            formated_prompt = self.template.format(**kwargs)
            num_tokens = self.llm_dict['llm'].get_num_tokens(formated_prompt)
            limit = 2048-1142
            while num_tokens > limit:
                if len(thoughts)>768:
                    kwargs['agent_scratchpad'] = thoughts = thoughts[:768]
                    formated_prompt = self.template.format(**kwargs)
                else:
                    formated_prompt = formated_prompt[:-1]
                num_tokens = self.llm_dict['llm'].get_num_tokens(formated_prompt)
                #print(f"WARNING: number of tokens : {num_tokens}.")
            return formated_prompt
    
    """
    react_prompt = ReACTPromptTemplate(
        llm_dict=llm_dict,
        template=react_template,
        tools=react_tools,
        # TODO : make some more elaborate profiles
        profile=angry_profile,
        input_variables=["input", "intermediate_steps"],
    )
    """

    react_prompt_with_context = ReACTPromptTemplate(
        llm_dict=llm_dict,
        template=react_template_with_context,
        tools=react_tools,
        # TODO : make some more elaborate profiles
        profile=angry_profile
        input_variables=["input", "intermediate_steps", 'history'] #"context"] 
    )
    

    # Output Parser:
    class ReACTOutputParser(AgentOutputParser):
        def parse(
            self,
            llm_output: str,
        ) -> Union[AgentAction, AgentFinish]:
            time.sleep(2)
            if "Final Answer:" in llm_output:
                finished = AgentFinish(
                    return_values={
                        "output":llm_output.split("Final Answer:")[-1].strip(),
                    },
                    log=llm_output,
                )
                return finished

            # Otherwise, we parse out the action and action input:
            regex = r"Action\s*\d*\s*:(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            if not match:
                #raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                error="Impossible to parse. Please use format above or express the Final Answer."
                observation="I think I know the final answer."
                raise OutputParserException(
                    error=observation, # error,
                    observation=observation,
                )
            action = match.group(1).strip()
            action_input = match.group(2)

            next_action = AgentAction(
                tool=action,
                tool_input= '"' + action_input.strip(" ").strip('"')+ '"',
                log=llm_output,
            )

            return next_action
    
    react_output_parser = ReACTOutputParser()
        
    # LLM Chains :
    react_chain = LLMChain(
        llm=llm,
        prompt=react_prompt_with_context,
    )

    react_tool_names = [tool.name for tool in react_tools]
    # Added to avoid the LLM to try to hallucinate its own observations...
    react_STOP_SEQs = ["\nObservation:", "Observation:"]
    
    react_agent = LLMSingleActionAgent(
        llm_chain=react_chain,
        output_parser=react_output_parser,
        stop=react_STOP_SEQs,
        allowed_tools=react_tool_names,
    )
    
    react_memory = ConversationBufferWindowMemory(k=3)

    react_agent_executor = ReACTAgentExecutor.from_agent_and_tools(
        agent=react_agent,
        tools=react_tools,
        verbose=True,
        memory=react_memory,
        # TODO : find a way to not have a limit...
        max_iterations=15,
        #early_stopping_method='generate',
    )
     
    wrappedAgent = WrappedAgent(
        agent_executor=react_agent_executor,
    )

    return wrappedAgent
