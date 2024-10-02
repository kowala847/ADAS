# Named tuple for holding task information
Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])

# Format instructions for FM response
FORMAT_INST = lambda request_keys: f"Reply EXACTLY with the following JSON format.\n{str(request_keys)}\nDO NOT MISS ANY FIELDS AND MAKE SURE THE JSON FORMAT IS CORRECT!\n"

# Description of the role of the FM Module
ROLE_DESC = lambda role: f"You are a {role}."

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(msg, model, system_message, temperature):
    """
    Function to get JSON response from GPT model.

    Args:
    - msg (str): The user message.
    - model (str): The model to use.
    - system_message (str): The system message.
    - temperature (float): Sampling temperature.

    Returns:
    - dict: The JSON response.
    """
    ...
    return json_dict

class FM_Module:
    """
    Base class for an FM module.

    Attributes:
    - output_fields (list): Fields expected in the output.
    - name (str): Name of the FM module.
    - role (str): Role description for the FM module.
    - model (str): Model to be used.
    - temperature (float): Sampling temperature.
    - id (str): Unique identifier for the FM module instance.
    """

    def __init__(self, output_fields: list, name: str, role='helpful assistant', model='gpt-3.5-turbo-0125', temperature=0.5) -> None:
        ...

    def generate_prompt(self, input_infos, instruction) -> str:
        """
        Generates a prompt for the FM.

        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.

        Returns:
        - tuple: System prompt and user prompt.

        An example of generated prompt:
        ""
        You are a helpful assistant.

        # Output Format:
        Reply EXACTLY with the following JSON format.
        ...

        # Your Task:
        You will given some number of paired example inputs and outputs. The outputs ...

        ### thinking #1 by Chain-of-Thought hkFo (yourself):
        ...

        # Instruction:
        Please think step by step and then solve the task by writing the code.
        ""
        """
        ...
        return system_prompt, prompt

    def query(self, input_infos: list, instruction, iteration_idx=-1) -> list[Info]:
        """
        Queries the FM with provided input information and instruction.

        Args:
        - input_infos (list): List of input information.
        - instruction (str): Instruction for the task.
        - iteration_idx (int): Iteration index for the task.

        Returns:
        - output_infos (list[Info]): Output information.
        """
        ...
        return output_infos

    def __repr__(self):
        return f"{self.agent_name}{self.id}"

    def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return self.query(input_infos, instruction, iteration_idx=iteration_idx)

class AgentSystem:
    def forward(self, taskInfo) -> Union[Info, str]:
        """
        Placeholder method for processing task information.

        Args:
        - taskInfo (Info): Task information.

        Returns:
        - Answer (Union[Info, str]): Your FINAL Answer. Return either a namedtuple Info or a string for the answer.
        """
        pass

def forward(self, taskInfo):
    # Instruction for initial reasoning
    cot_initial_instruction = "Please think step by step and then solve the task."

    # Instruction for reflecting on previous attempts and feedback to improve
    cot_reflect_instruction = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_module = FM_Module(['thinking', 'answer'], 'Chain-of-Thought')

    # Instruction for providing feedback and correcting the answer
    critic_instruction = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output 'True' in 'correct'."
    critic_module = FM_Module(['feedback', 'correct'], 'Critic')

    N_max = 5 # Maximum number of attempts

    # Initial attempt
    cot_inputs = [taskInfo]
    thinking, answer = cot_module(cot_inputs, cot_initial_instruction, 0)

    for i in range(N_max):
        # Get feedback and correct status from the critic
        feedback, correct = critic_module([taskInfo, thinking, answer], critic_instruction, i)
        if correct.content == 'True':
            break

        # Add feedback to the inputs for the next iteration
        cot_inputs.extend([thinking, answer, feedback])

        # Reflect on previous attemps and refine the answer
        thinking, answer = cot_module(cot_inputs, cot_reflect_instruction, i + 1)
    return answer
