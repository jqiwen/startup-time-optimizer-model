# **Adaptive Container Startup Optimization (RL-based)**

This project implements an **adaptive container startup optimization system** that applies a Reinforcement Learning (RL) model to automatically tune container configuration parameters with the goal of **minimizing container startup latency**.

The optimized deployment is designed to operate within a **CI/CD pipeline on IBM Cloud**, enabling continuous delivery of improved configurations while ensuring automation, reproducibility, and seamless updates.

---

## **Installation**

```bash
pip install pyyaml
```
---

## **Execute**

```bash
python optimizer.py
```
then follow the step on popup window

### **Step 1: Generate Local Environment for Your Projec**

1. The output location is fixed, it will always be generated under `./local_env/local_app`
2. Click `Browse`, select your original project directory.
3. Click `Build Local Environment`. The process will run in a terminal window. Once the build completes, a success notification will appear.
4. After the environment is successfully created, click **`Next`** to continue to Step 2.

### **Second, parse your deploy yaml file**
1. Click the top button, Select the YAML file from your project (e.g., your deployment or compose file).
5. Specify the resource fields you want to extract (e.g., cpu,memory).
6. The tool automatically parses the configuration values such as limits and reservations.
7. Click Save and Exit to store the parsed results.
The parsed resource configuration is then used as input to the RL model.

RL Optimization

