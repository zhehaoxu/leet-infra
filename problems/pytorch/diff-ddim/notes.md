# 详解：DDIM 确定性去噪一步

## 出处
Song et al., *Denoising Diffusion Implicit Models*（arXiv:2010.02502）式 (12)：\(x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta+\sigma_t z\)。取 \(\sigma_t=0\) 即本题的确定性更新（\(\epsilon_\theta\) 是模型预测的噪声，\(\hat x_0\) 由其反解）。

## 核心与易错点
1. **先反解 \(\hat x_0\) 再前跳**：\(\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}\)，然后按 \(\bar\alpha_{t-1}\) 重新组合。两步缺一不可、方向不能反。
2. **\(\sigma_t=0\) 无噪声**：与 DDPM 后验不同，DDIM 确定性版不注入 \(z\)，同一输入必得同一输出，这是它能少步采样与做「反演」的原因。
3. **恒等/全去噪退化**：\(\bar\alpha_{t-1}=\bar\alpha_t\) 应还原成 \(x_t\)；\(\bar\alpha_{t-1}=1\) 应等于 \(\hat x_0\)。这两条是判公式是否写对的探针。
4. 分母 \(\sqrt{\bar\alpha_t}\)，\(\bar\alpha_t\in(0,1]\) 不会为零。

## 对拍口径
参考解向量化两步。判题 `_oracle` 逐元素双循环用 `math.sqrt` 重算 \(\hat x_0\) 与 \(x_{t-1}\)；另用「前向生成 → DDIM 反解」验证 \(\hat x_0\) 还原、\(\bar\alpha_{t-1}=\bar\alpha_t\) 恒等、\(\bar\alpha_{t-1}=1\) 回到 \(\hat x_0\)（结构不同、数值等价），`allclose(atol=1e-5,rtol=1e-5)`。

## 复杂度
逐元素 \(O(N)\)。
