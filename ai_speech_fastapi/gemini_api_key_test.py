import google.generativeai as genai

genai.configure(api_key="AIzaSyD_MvgwSiQ__pQCb2JRWWqGSx-2XrcFv6A")
model = genai.GenerativeModel("gemini-1.5-pro")
response = model.generate_content("Tell me a joke")
print(response.text)
