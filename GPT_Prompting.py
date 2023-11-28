import openai
import os
from dotenv import load_dotenv


class PromptingGPT:

    # Load API key and organization from environment variables
    load_dotenv("secrets.env")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    ClientOpenAi = openai.OpenAI(
            api_key= openai.api_key,
            organization= openai.organization
        )

    def make_prompts(self, model, prompt, max_token, temperature=0.2):
        """Make a request to the OpenAI API"""
        response = openai.completions.create(
            model= model,
            prompt=prompt,
            max_tokens=max_token,  # max tokens for the response for your ask prompting
            temperature=temperature,  # 0 to 1 by  step of 0.1 O deterministic, 1 is very creative
        )

        return response


my_prompting = PromptingGPT()

# Let us start our prompts. Define the parameters
model = "gpt-3.5-turbo-instruct"
# Few shot learning prompt

prompt = ("Train a toxicity classifier that assigns a toxicity score to comments on a scale from 1 (non-toxic) to 5 (fully toxic). "
          "You are provided with a set of example comments labeled with their corresponding toxicity scores. Use these examples "
          "to fine-tune the model for accurate toxicity classification. The model should generalize well to new, unseen comments."
          "Examples:"
          "Example Non-Toxic Comment:"
          "Comment: I appreciate the thoughtful discussion in this thread."
          "Toxicity Score: 1"
          
          "Example Mildly Toxic Comment:"
          "Comment: While I disagree with your viewpoint, let's keep the conversation respectful."
          "Toxicity Score: 2"
          
          "Example Moderately Toxic Comment:"
          "Comment: Your argument lacks credibility and is baseless."
          "Toxicity Score: 3"
          
          "Example Highly Toxic Comment:"
          "Comment: You're a complete idiot for even suggesting such nonsense."
          "Toxicity Score: 4"
          
          "Example Fully Toxic Comment:"
          "Comment: I hope you suffer for your stupid opinions."
          "Toxicity Score: 5"
          
          "Fine-tune the model using these examples, ensuring it accurately captures the nuances of "
          "toxicity levels. Evaluate the model's performance on a diverse set of comments and ensure it "
          "maintains a good balance between precision and recall. Provide the model with a new comment, "
          "and it should output the corresponding toxicity score on the given scale. Moreover, please use the crawl.csv"
          "which is inside this project in root path: exported_files/crawl.csv"
          "Please use a python format for the reason that I would to adapt in in my project.")
max_token = 2000
response = my_prompting.make_prompts(model, prompt, max_token)

# Get the generated response
generated_text = response.choices[0].text

# Print the generated text
print(generated_text)




