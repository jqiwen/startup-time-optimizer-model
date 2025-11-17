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
Follow the 2 steps on popup window to start model training: 

### **Step 1: Generate Local Environment for Your Projec**

1. The output location is fixed, it will always be generated under `./local_env/local_app`
2. Click **`Browse`**, select your original project directory.
3. Click **`Build Local Environment`**. The process will run in a terminal window. Once the build completes, a success notification will appear.
4. After the environment is successfully created, click **`Next`** to continue to Step 2.

### **Step 2: Parse Your Deployment YAML File**
1. Click button **`Select YAML File`** on the top, choose a deployment-related YAML file (e.g., Docker Compose file).
2. Enter the resource fields that can be optimized, and use **`;`** to separate (e.g., `cpu;memory`).
3. Click **`PARSE`**, the tool will automatically extract resource configurations such as **limits** and **reservations** for each service. You may check the result on the context board below.
4. Click **`Save and Start Training`**, the parsed results will be saved to `./local_env/yaml_parser_results.json`.

### **Step 3: Update local Docker Compose**

### **Step 4: Start RL training on local environment**

### **Step 5: Review Results and Decide Whether to Apply Changes**


---

### **RL Model Training**

Once the training start, it will generate a local docker componser first, that includes ... for local envirenoment.
Then based on the knowledge base, follow the MAKP-K, RL learn.

Once training completes, the optimized configuration will be displayed. The suggested reservation of CPU, memory, and compare with the original performance.
User can choose if want to apply the change on their original project
