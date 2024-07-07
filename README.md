<h1 style="font-family: Arial; font-size: 24px;">DCAN——Deep Covariance Alignment Network</h1>

<p style="font-family: Verdana; font-size: 14px;">We propose a transfer learning method called Deep Covariance Alignment Network (DCAN) for cross-linguistic depression detection using audio data, where the source domain is English and the target domain is Chinese.</p>
<p style="font-family: Verdana; font-size: 14px;">"Our data input consists of three types:</p>

<ul style="font-family: Verdana; font-size: 14px;">
<li>Manually extracted features</li>
<li>Features extracted using CNN AE</li>
<li>Downsampled audio data</li>
</ul>

<p style="font-family: Verdana; font-size: 14px;">The code is stored in the 'extract_features' folder."</p>

<p style="font-family: Verdana; font-size: 14px;">Our Chinese dataset uses the MODMA dataset, which can be downloaded from <a href='https://modma.lzu.edu.cn/data/index/'>https://modma.lzu.edu.cn/data/index/</a></p>
<p style="font-family: Verdana; font-size: 14px;">Our English dataset uses the DAIC-WOZ dataset, which can be downloaded from <a href='https://dcapswoz.ict.usc.edu/'>https://dcapswoz.ict.usc.edu/</a></p>

<p style="font-family: Verdana; font-size: 14px;">In addition, the data preprocessing and model construction parts of our CNN AE code reference <a href='https://github.com/SaraS92/CAE_ADD'>https://github.com/SaraS92/CAE_ADD</a>.</p>
<p style="font-family: Verdana; font-size: 14px;">Furthermore, parts of our transfer learning code are referenced from <a href='https://github.com/jindongwang/transferlearning'>https://github.com/jindongwang/transferlearning</a>.</p>
