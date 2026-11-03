# License

vault-scene-tagger is free software for personal use: run it, read it, modify it for your own
libraries, and send improvements back. You may not sell it or bundle it into a product you sell.

It is provided as is, without warranty of any kind. It only ever writes `*.scenes.json` sidecar
files next to your videos and never modifies, moves or deletes a video.

## The scene model

The fine-tuned scene model (`vault-r<N>`) that the `phar` engine can load is **not** part of this
repository. It is distributed to Vault Pro supporters through the Vault app and is licensed for
use with Vault and this tool on their own libraries; redistributing the weights is not allowed.
The model is a fine-tune of [P-HAR](https://github.com/rlleshi/phar)'s TimeSformer (Apache-2.0).
P-HAR's original weights, which this tool can also run, are Apache-2.0 and come from P-HAR's own
releases.
