import VideoThumb from "@/public/images/hero-image-01.jpg";

export default function HeroHome() {
  return (
    <section>
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        {/*content */}
        <div className="py-12 md:py-20">
          {/* Section header */}
          <div className="pb-12 text-center md:pb-20">
            <h1
              className="animate-[gradient_6s_linear_infinite] bg-[linear-gradient(to_right,var(--color-gray-200),var(--color-indigo-200),var(--color-gray-50),var(--color-indigo-300),var(--color-gray-200))] bg-[length:200%_auto] bg-clip-text pb-5 font-nacelle text-4xl font-semibold text-transparent md:text-5xl"
              data-aos="fade-up"
            >
              Welcome to Deepfake Detection System!
            </h1>
            <div className="mx-auto max-w-3xl">
              <p className="mb-8 text-3xl text-indigo-200/60">
                We have created an advanced AI-powered system to detect manipulated audio and video content. Techniques include analyzing inconsistencies in facial expressions, lighting, and audio. Machine learning models detect artifacts, pixel mismatches, and unnatural movements.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
