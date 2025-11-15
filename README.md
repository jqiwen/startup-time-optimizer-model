# **Adaptive Container Startup Optimization (RL-based)**

This project implements an **adaptive container startup optimization system** that applies a Reinforcement Learning (RL) model to automatically tune container configuration parameters with the goal of **minimizing container startup latency**.

The optimized deployment is designed to operate within a **CI/CD pipeline on IBM Cloud**, enabling continuous delivery of improved configurations while ensuring automation, reproducibility, and seamless updates.

---

## **YAML Resource Parser (Preprocessing Tool)**

### **Installation**

```bash
pip install pyyaml
```


1. Navigate to the models/ directory.
2. Run: python parse.py
3. A pop-up window will appear.
4. Select the YAML file from your project (e.g., your deployment or compose file).
5. Specify the resource fields you want to extract (e.g., cpu,memory).
6. The tool automatically parses the configuration values such as limits and reservations.
7. Click Save and Exit to store the parsed results.
The parsed resource configuration is then used as input to the RL model.

RL Optimization

