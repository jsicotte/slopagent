from ollama import chat
from ollama import ChatResponse

class Agent:
    def get_user_message() -> str:
        input_response = input("")

        return input_response

if __name__ == "__main__":
    agent = Agent()
    conversation = []
    while True:
        print("You")
        user_maessage = agent.get_user_message()
        conversation.append(user_maessage)
        inference_response = chat(model="gemma4:e4b-mlx", messages=conversation)
        conversation.append(inference_response["message"]["content"])

        print("LLM")
        print(inference_response)




# func main() {
# 	client := anthropic.NewClient()

# 	scanner := bufio.NewScanner(os.Stdin)
# 	getUserMessage := func() (string, bool) {
# 		if !scanner.Scan() {
# 			return "", false
# 		}
# 		return scanner.Text(), true
# 	}

# 	agent := NewAgent(&client, getUserMessage)
# 	err := agent.Run(context.TODO())
# 	if err != nil {
# 		fmt.Printf("Error: %s\n", err.Error())
# 	}
# }



# type Agent struct {
# 	client         *anthropic.Client
# 	getUserMessage func() (string, bool)
# }