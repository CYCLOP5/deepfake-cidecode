export const metadata = {
  title: "Home - Open PRO",
  description: "Page description",
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
