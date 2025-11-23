# **Adaptive Container Startup Optimization (RL-based)**

This project implements an **adaptive container startup optimization system** that applies a Reinforcement Learning (RL) model to automatically tune container configuration parameters with the goal of **minimizing container startup latency**.

The optimized deployment is designed to operate within a **CI/CD pipeline on IBM Cloud**, enabling continuous delivery of improved configurations while ensuring automation, reproducibility, and seamless updates.

---

## **Installation**

Make sure your docking is running, and install the following packages
```bash
pip install pyyaml
pip install stable-baselines3[extra]
pip install gymnasium
pip install pandas numpy requests matplotlib
```
Build a local image
```bash
cd local_env\sidecar
docker build -t project-sidecar:latest .
```
---

## **Execution**

```bash
python optimizer.py
```
Follow the 2 steps on popup window to prepare the model training: 

### **Step 1: Generate Local Environment for Your Projec**

1. The output location is fixed, it will always be generated under `./local_env/local_app`
2. Click **`Browse`**, select your original project directory.
3. Click **`Build Local Environment`**. The process will run in a terminal window. Once the build completes, a success notification will appear.
4. After the environment is successfully created, click **`Next`** to continue to Step 2.

### **Step 2: Input your original yaml resource File**
1. input your original reservation and limit for CPU, Memory, and Heap
2. Click `Generate JSON` to check corretness
3. Click **`Save and Run`**, local docker will be updated automatically.


``` bash
python train.py
```

### **Step 4: Start RL training on local environment**

### **Step 5: Review Results and Decide Whether to Apply Changes**


---

### **RL Model Training**

Once the training start, it will generate a local docker componser first, that includes ... for local envirenoment.
Then based on the knowledge base, follow the MAKP-K, RL learn.

Once training completes, the optimized configuration will be displayed. The suggested reservation of CPU, memory, and compare with the original performance.
User can choose if want to apply the change on their original project
