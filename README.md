
# Covenant Labs SDK

**An experimental LLM encryption library**

The Covenant Labs SDK is a proof-of-concept project exploring the use of end-to-end encryption for **language models**. It is not intended for production use **yet**. This library is designed to interface with the [Covenant Labs](https://covenantlabs.ai/) platform, allowing users to experiment with deploying encrypted models and test various ideas.

⚠ **Warning:** This library is experimental. USE AT YOUR OWN RISK.

---
## Installation
```bash
pip install "git+https://github.com/Covenant-Labs-AI/covenant_labs_sdk.git"

```
---
## Usage
```python
client = CovenantClient(
    deployment_key="YOUR-PUBLIC-DEPLOYMENT-KEY", 
    secret_key="YOUR-PRIVATE-SECRET-KEY",
)
client.encrypt_model()
client.deploy()
# EX: llama 3.2 1b
messages = [
    {
        "role": "system",
        "content": (
            "You are an AI that extracts structured data from user messages. "
            "Your output must always be in JSON format with the fields: "
            "`greeting`, `mood`, and `name` (if any). "
            "If a field isn't present, set it to null."
        ),
    },
    {
        "role": "user",
        "content": "Hello, how are you? My name is Sarah and im happy.",
    },
]

data = client.secure_inference(messages)

print(data)
```
## MEP Model Encryption Protocol

The **MEP (Model Encryption Protocol)** is a key concept introduced in this SDK to explore encrypting models with the goal of protecting sensitive model data, ensuring privacy, and enabling secure communication between users and models.

### Core Ideas of MEP:
1. **End-to-End Encryption**: Encryption is applied directly to the language model’s parameters and weights, ensuring that data remains protected both during transit and at rest.
   
2. **Model Privacy**: The protocol focuses on maintaining the privacy of the model during interaction, limiting access to model internals while still allowing users to interact with it securely.

3. **Secure Model Deployment**: Users can deploy encrypted models on the Covenant Labs platform while maintaining full control over the encryption keys, ensuring that the model’s integrity is preserved.

4. **Privacy-Preserving Tinkering**: MEP allows users to experiment with encrypted models while ensuring no unauthorized access or modification occurs.

By using MEP, developers can explore new possibilities for privacy and security in the realm of large language models, especially for applications in sensitive industries.

## Features

- Simple and flexible API for rapid experimentation
- Easily extendable for various use cases
- Promotes creative exploration and tinkering


## Contributing, Feedback, and Testing

We’re excited to have you involved in the development of Covenant Labs SDK! As an experimental and unique project, your feedback and contributions are invaluable to making it better.

### How You Can Contribute

1. **Testing**: As the SDK is in an experimental phase, testing is crucial. Try out different encryption concepts, models, and integrations, and report any issues or inconsistencies. Your testing helps improve the platform and ensures its reliability.

2. **Feedback**: Let us know what you think! Share your experience, ideas, or any challenges you encounter while using the SDK. Whether it’s about the encryption mechanisms, user interface, or performance, all feedback is welcome.

3. **Contributing Code**: If you have an improvement, bug fix, or new feature in mind, we welcome your pull requests! To contribute code, please follow these steps:
   - **Fork the repository** and create a branch for your changes.
   - **Make your changes**, keeping the code clean, modular, and well-commented.
   - **Test your changes** thoroughly to ensure everything works as expected.
   - **Submit a pull request** with a detailed description of your changes, including any related issues or ideas.

4. **Identifying and Reporting Security Vulnerabilities**:  
   Security is a top priority in this project. Given that we’re working with encryption and sensitive data, it’s important that we identify and fix potential security vulnerabilities early. If you discover any vulnerabilities, please:
   - **Do not disclose them publicly** to prevent exploitation.
   - **Report the vulnerability privately** to the project maintainers using a direct message or email (contact details will be provided upon request).
   - **Provide as much detail as possible** so we can assess the impact and work on a fix quickly.
   
   Your help in identifying vulnerabilities is crucial for maintaining the integrity and security of the SDK.

This is a **collaborative and experimental** project, and we appreciate all the help we can get to shape it into something amazing!


#### "Not your models, not your mind."
---

