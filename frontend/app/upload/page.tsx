"use client";

import { useState } from "react";
import Link from "next/link";

export default function VideoUpload() {
	const [file, setFile] = useState<File | null>(null);
	const [message, setMessage] = useState<string>("");
	const [loading, setLoading] = useState<boolean>(false);
	const [showResults, setShowResults] = useState<boolean>(false);

	const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		if (event.target.files) {
			setFile(event.target.files[0]);
		}
	};

	const handleUpload = async () => {
		if (!file) return;

		setLoading(true);
		setMessage("");
		setShowResults(false); // Hide results button on new upload

		const formData = new FormData();
		formData.append("file", file);

		try {
			const response = await fetch("http://localhost:5000/upload", {
				method: "POST",
				body: formData,
			});

			const data = await response.json();
			if (data.error) {
				setMessage(data.error);
			} else {
				setMessage("File uploaded successfully!");
				setShowResults(true); // Show results button after success
			}
		} catch (error) {
			setMessage("Error uploading file.");
		} finally {
			setLoading(false);
		}
	};

	return (
		<section className="bg-gray-900 min-h-screen flex items-center justify-center">
			<div className="mx-auto max-w-6xl px-4 sm:px-6">
				<div className="py-12 md:py-20 text-center">
					<div className="pb-12">
						<h1 className="text-3xl font-semibold text-white md:text-4xl">
							Please upload a video for detection
						</h1>
					</div>
					<form
						onSubmit={(e) => {
							e.preventDefault();
							handleUpload();
						}}
						className="mx-auto max-w-[400px]"
					>
						<div className="space-y-5">
							<label className="cursor-pointer flex flex-col items-center p-6 border-2 border-dashed rounded-lg border-gray-700">
								<div className="w-10 h-10 mb-2 text-gray-500">
									📂
								</div>
								<span className="text-gray-400">
									Click to upload
								</span>
								<input
									type="file"
									accept="video/*, image/*, audio/*"
									className="hidden"
									onChange={handleFileChange}
								/>
							</label>
							{file && (
								<p className="mt-2 text-sm text-gray-400">
									Selected: {file.name}
								</p>
							)}
						</div>
						<div className="mt-6">
							<button
								type="submit"
								className="w-full bg-indigo-600 text-white py-2 rounded-lg hover:bg-indigo-500 transition-colors duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
								disabled={!file || loading}
							>
								{loading ? "Uploading..." : "Submit"}
							</button>
						</div>
					</form>
					{loading && (
						<div className="mt-4">
							<div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
							<p className="text-gray-400 mt-2">
								Processing file...
							</p>
						</div>
					)}
					{message && (
						<p className="mt-4 text-gray-400">{message}</p>
					)}
					{showResults && (
						<div className="mt-6 text-center">
							<Link
								className="inline-block px-6 py-3 text-lg font-medium text-[#030712] bg-white rounded-lg hover:opacity-80 transition duration-300"
								href="http://localhost:5000/report"
							>
								Show results!
							</Link>
						</div>
					)}
				</div>
			</div>
		</section>
	);
}
