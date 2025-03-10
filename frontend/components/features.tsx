import Image from "next/image";
import BlurredShapeGray from "@/public/images/blurred-shape-gray.svg";
import BlurredShape from "@/public/images/blurred-shape.svg";
import FeaturesImage from "@/public/images/features.png";

export default function Features() {
  return (
    <section className="relative">
      <div
        className="pointer-events-none absolute left-1/2 top-0 -z-10 -mt-20 -translate-x-1/2"
        aria-hidden="true">
      </div>
      <div
        className="pointer-events-none absolute bottom-0 left-1/2 -z-10 -mb-80 -translate-x-[120%] opacity-50"
        aria-hidden="true">
      </div>
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <div className="border-t py-12 [border-image:linear-gradient(to_right,transparent,--theme(--color-slate-400/.25),transparent)1] md:py-20">
          {/* Section header */}
          <div className="mx-auto max-w-3xl pb-4 text-center md:pb-12">
            <div className="inline-flex items-center gap-3 pb-3 before:h-px before:w-8 before:bg-linear-to-r before:from-transparent before:to-indigo-200/50 after:h-px after:w-8 after:bg-linear-to-l after:from-transparent after:to-indigo-200/50">
            </div>
            <h2 className="animate-[gradient_6s_linear_infinite] bg-[linear-gradient(to_right,var(--color-gray-200),var(--color-indigo-200),var(--color-gray-50),var(--color-indigo-300),var(--color-gray-200))] bg-[length:200%_auto] bg-clip-text pb-4 font-nacelle text-3xl font-semibold text-transparent md:text-4xl">
              Built for CCITR
            </h2>
            <p className="text-lg text-indigo-200/65">
              Our entire model can be explained generally by the following image.
            </p>
          </div>
          <div className="flex justify-center pb-4 md:pb-12" data-aos="fade-up">
            <Image
              className="max-w-none"
              src={FeaturesImage}
              width={1104}
              height={384}
              alt="Features"
            />
          </div>
          {/* Items */}
          <div className="mx-auto grid max-w-sm gap-12 sm:max-w-none sm:grid-cols-2 md:gap-x-14 md:gap-y-16 lg:grid-cols-3">
            <article>
              <svg
                className="mb-3 fill-indigo-500"
                xmlns="http://www.w3.org/2000/svg"
                width={24}
                height={24}
              >
                <path d="M0 0h14v17H0V0Zm2 2v13h10V2H2Z" />
                <path
                  fillOpacity=".48"
                  d="m16.295 5.393 7.528 2.034-4.436 16.412L5.87 20.185l.522-1.93 11.585 3.132 3.392-12.55-5.597-1.514.522-1.93Z"
                />
              </svg>
              <h3 className="mb-2 text-lg font-bold text-gray-100 tracking-wide">
  <span className="text-indigo-400">◆</span> Models
</h3>
<div className="text-indigo-200/75 space-y-2">
  <p>
    <span className="font-semibold text-indigo-300">Frames-Based Detection</span> – Uses <span className="text-indigo-400">EfficientNet-B0</span>, fine-tuned for binary classification (Real vs. Fake). Detected faces are resized, normalized, and classified. <span className="italic">Grad-CAM</span> provides visual explanations.
  </p>
  <p>
    <span className="font-semibold text-indigo-300">MRI-GAN Detection</span> – Generates MRI-like images using a <span className="text-indigo-400">U-Net-based GAN</span> with <span className="italic">PatchGAN</span> discriminator. Loss combines <span className="text-indigo-400">cGAN</span>, L2, and SSIM-based perceptual loss (<span className="italic">τ = 0.15</span>). The generated MRI images are classified using <span className="text-indigo-400">EfficientNet-B0</span>.
  </p>
</div>

            </article>
            <article>
              <svg
                className="mb-3 fill-indigo-500"
                xmlns="http://www.w3.org/2000/svg"
                width={24}
                height={24}
              >
                <path fillOpacity=".48" d="M7 8V0H5v8h2Zm12 16v-4h-2v4h2Z" />
                <path d="M19 6H0v2h17v8H7v-6H5v8h19v-2h-5V6Z" />
              </svg>
              <h3 className="mb-1 font-nacelle text-[1rem] font-semibold text-gray-200">
  Face Detection & Data Augmentation
</h3>
<p className="text-indigo-200/65">
  MTCNN detects faces in sampled frames (every 10th) to reduce computation. 
  Undetected frames/videos are discarded. Preprocessing includes augmentations 
  like rotation, scaling, flipping, and distractions to improve robustness.
</p>

            </article>
            <article>
              <svg
                className="mb-3 fill-indigo-500"
                xmlns="http://www.w3.org/2000/svg"
                width={24}
                height={24}
              >
                <path d="M23.414 6 18 .586 16.586 2l3 3H7a6 6 0 0 0-6 6h2a4 4 0 0 1 4-4h12.586l-3 3L18 11.414 23.414 6Z" />
                <path
                  fillOpacity=".48"
                  d="M13.01 12.508a2.5 2.5 0 0 0-3.502.482L1.797 23.16.203 21.952l7.71-10.17a4.5 4.5 0 1 1 7.172 5.437l-4.84 6.386-1.594-1.209 4.841-6.385a2.5 2.5 0 0 0-.482-3.503Z"
                />
              </svg>
              <h3 className="mb-1 font-nacelle text-[1rem] font-semibold text-gray-200">
  Audio Detection Workflow
</h3>
<p className="text-indigo-200/65">
  FoR dataset is used. Librosa loads audio, skipping silent parts. Volume is normalized, 
  lengths are standardized, and mel-spectrograms are flattened for training.  
  Models tested: RFC, Logistic Regression, XGBoost. Logistic Regression was chosen 
  for best balance without overfitting.
</p>

            </article>
          </div>
        </div>
      </div>
    </section>
  );
}
