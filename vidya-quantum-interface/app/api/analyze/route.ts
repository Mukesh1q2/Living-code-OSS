import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const body = await req.json();
  const { text } = body as { text: string };

  // Call your real Sanskrit Engine here
  // const resp = await fetch(process.env.SANSKRIT_ENGINE_URL!, { method: "POST", body: JSON.stringify({ text }) });
  // const data = await resp.json();

  // Mock response structure aligning with your capabilities
  const data = {
    input: text,
    tokens: ["रामः", "वनम्", "गच्छति"],
    analysis: [
      {
        token: "रामः",
        lemma: "राम",
        pos: "NOUN",
        case: "NOM",
        number: "SG",
        gender: "M",
      },
      {
        token: "वनम्",
        lemma: "वन",
        pos: "NOUN",
        case: "ACC",
        number: "SG",
        gender: "N",
      },
      {
        token: "गच्छति",
        lemma: "गम्",
        pos: "VERB",
        person: 3,
        number: "SG",
        tense: "PRS",
      },
    ],
    rulesFired: ["सुप्-प्रत्यय", "तिङ्-प्रत्यय", "विभक्ति-निर्णय"],
  };

  return NextResponse.json(data);
}