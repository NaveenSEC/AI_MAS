## Research Papers & Resources

### LLM - Visaul Explainers
- **Jay Alammar**, "**The Illustrated Transformer**" (jalammar.github.io/illustrated-transformer). This is the best place to start. Almost every step is a diagram: how words become numbers, how attention works, and how the encoder and decoder fit together. University courses use it widely.
    - **Book Reference**
        - **Hands-On Large Language Models by Jay Alammar & Maarten Grootendorst**
- Jay Alammar, "**The Illustrated GPT-2**" (jalammar.github.io/illustrated-gpt2). The follow-up shows how a ChatGPT-style model predicts the next word, one token at a time. This is the core of "how an LLM works."
- **Transformer Explainer** (poloclub.github.io/transformer-explainer, Georgia Tech, arXiv:2408.04619). You type a sentence and watch a real GPT-2 process it live in your browser. You can see the numbers moving through the model.
3- **Blue1Brown, "But what is a GPT?"** and the attention video that follows it (YouTube). These are animated and very intuitive, about 30 minutes each.
- Vaswani et al., **"Attention Is All You Need"** (2017), arXiv:1706.03762. This is the original Transformer paper. Read it after items 1–4. Figure 1 is the architecture diagram everyone reuses.
- Ouyang et al., **"InstructGPT"** (2022), arXiv:2203.02155. Figure 2 shows in three pictures how a raw LLM becomes a helpful assistant: supervised fine-tuning, a reward model, then reinforcement learning from human feedback (RLHF).
- Zhao et al., **"A Survey of Large Language Models,"** arXiv:2303.18223. This is the long reference survey. Use it to look up details, not to read cover to cover.

#### Reading Tips
- Spend two days on items 1, 2 and 4 to build intuition, then read the Minaee survey. 
- Item 5 will then be easy, and item 6 explains why chat models follow instructions. 
- That last point matters for your topic, because prompt injection exploits exactly that instruction-following.

### Large Language Models: A Survey
- Authors: Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, Jianfeng Gao (Led by the RUCAIBox team)
- University: Renmin University of China (Gaoling School of Artificial Intelligence)
- Gitrepo: https://github.com/RUCAIBox/LLMSurvey
- Reference Points:
    - TODO

### SoK: When Safe Agents Fail Together: The Security of Multi Agent LLM Systems
- Authors: Rui Yang, Junjie Xu, Zhengyu Liu, Neil Fendley, Yang Hong, Ziyang Li, Yinzhi Cao, Johns Hopkins
- University: Johns Hopkins University (and Nanyang Technological University)
- Journal: 
- Gitrepo: Not publicly released (arXiv:2609.00595)
- Reference Points: 
    - TODO

### On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents
- Authors: Jen-tse Huang, Jiaxu Zhou, Tailin Jin, Xuhui Zhou, Zixi Chen, Wenxuan Wang, Youliang Yuan, Michael R. Lyu, Maarten Sap
- University: Chinese University of Hong Kong (CUHK), Carnegie Mellon University, Tsinghua University, Peking University, Renmin University of China
- Journal:
- Gitrepo: https://github.com/CUHK-ARISE/MAS-Resilience
- Reference Points: 
    - TODO

### An Overview of Security Threats, Attack Detection and Defense for Large-Scale Multi-Agent Systems (LSMAS) in Internet of Things (IoT)
- Authors: Jijing Cai, Long Wen, Hailin Feng, Kai Fang, Junxin Chen, Wei Wei
- University: Xi'an University of Technology / Henan Polytechnic University (associated institutions for this author group)
- Journal: 
- Gitrepo: Not publicly released (IEEE Transactions on Industrial Cyber-Physical Systems)
- Reference Points:
    - TODO

### Template: G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems
- Authors: Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, Yang Wang
- University: University of Science and Technology of China (USTC), Tongji University, Wuhan University
- Journal: 
- Gitrepo: https://github.com/wslong20/G-safeguard
- Reference Points:
    - TODO

### Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation (ICML 2026)
- Authors: Chengcan Wu, Zhixin Zhang, Mingqian Xu, Zeming Wei, and Meng Sun
- University: Peking University / State Key Laboratory of Software Development Environment (Beihang University) (commonly associated authors/institutions for this work)
- Journal: 
- Gitrepo: https://github.com/ChengcanWu/BPD
- Reference Points:
    - TODO

### Topology Linearization for Multi-Agent Systems Security: Mitigating Malicious Propagation via Path Decomposition
- Authors: Yan Liu, Sujie Shao, Shaoyong Guo, Chenyu Wang, Chao Yang, Zhibin Zang
- University:
- Journal: 
- Gitrepo:
- Referemce Points:
    - Collaborative task-solving leads to **explosive malicious propagation**
    - LLM-based multi-agent architectures, agents communicate across high-connectivity - complex interaction graphs (such as star, mesh, or fully connected topologies
    - **Key Components of the Framework**\
    Chain-Type Secure Path Reconstruction Method (C-SPRM)\
    Spatio-Temporal Attack Surface (STAS) Minimization\
    Graph-to-Sequence Path Decomposition via Reinforcement Learning (G2S-PDR)\
    Adaptive Topology Virtualization\
    - **Key Benefits**\
    Reduced Blast Radius\
    Exponentially Higher Attack Difficulty\
    Controllable Overhead\

### Template: Title...
- Authors:
- University:
- Gitrepo:
- Reference Points:
    - TODO

### Summary
- Backward Propogation Detection (BPD)
- Topology Linearation -> To mitigate malicious propagation via path decomposition

### AI-Attacks
- Prompt Injection
- Data Poisioning
- Memory Poison
- Tool Attack

### Others 
- Lateral threat Movement
- Hallucinations