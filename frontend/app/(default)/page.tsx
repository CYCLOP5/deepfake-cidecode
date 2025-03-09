export const metadata = {
	title: "DeepFake Detector",
	description: "DeepFake Detection System",
};

import PageIllustration from "@/components/page-illustration";
import Hero from "@/components/hero-home";
import Features from "@/components/features";
import Cta from "@/components/cta";

export default function Home() {
	return (
		<>
			<Hero />
			<Features />
			<Cta />
		</>
	);
}
