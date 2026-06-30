# from openai import OpenAI, RateLimitError, AuthenticationError

# client = OpenAI()

# import os
# print("env key tail:", (os.environ.get("OPENAI_API_KEY") or "NONE")[-6:])
# print("client key tail:", client.api_key[-6:])

# try:
#     response = client.responses.create(
#         model="gpt-4.1-2025-04-14",
#         input="Reply with exactly: API works"
#     )

#     print("SUCCESS")
#     print(response.output_text)

# except AuthenticationError as e:
#     print("AUTH ERROR: Your API key is missing, wrong, or not loaded.")
#     print(e)

# except RateLimitError as e:
#     print("RATE LIMIT ERROR: The key works, but your org/project is capped right now.")
#     print(e)

# except Exception as e:
#     print("OTHER ERROR:")
#     print(e)



# from openai import OpenAI
# client = OpenAI()
# for m in client.models.list().data:
#     if "gpt-4.1" in m.id:
#         print(m.id)

print("Available models:")
