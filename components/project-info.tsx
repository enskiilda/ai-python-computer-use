import { motion } from "motion/react";
import { VercelIcon } from "./icons";
import { ComputerIcon } from "lucide-react";
import Link from "next/link";

export const ProjectInfo = () => {
  return (
    <motion.div className="w-full px-4">
      <div className="rounded-lg border-border border p-6 flex flex-col gap-4 text-center text-base dark:text-zinc-400">
        <p className="flex flex-row justify-center gap-4 items-center text-zinc-900 dark:text-zinc-50">
          <VercelIcon size={16} />
          <span>+</span>
          <ComputerIcon />
        </p>
        <h3 className="text-center text-2xl font-bold">Computer Use Agent</h3>
        <p>
          This demo showcases a Computer Use Agent built with{" "}
          <StyledLink href="https://onkernel.com">Kernel SDK</StyledLink>,{" "}
          <StyledLink href="https://build.nvidia.com/meta/llama-4-maverick-17b-128e-instruct">
            NVIDIA Llama 4
          </StyledLink>
          , and real-time WebSocket streaming.
        </p>
        <p>
          The AI controls a browser in real-time using computer tools for
          mouse, keyboard, and screenshot actions.
        </p>
      </div>
    </motion.div>
  );
};

const StyledLink = ({
  children,
  href,
}: {
  children: React.ReactNode;
  href: string;
}) => {
  return (
    <Link
      className="text-blue-500 dark:text-blue-400"
      href={href}
      target="_blank"
    >
      {children}
    </Link>
  );
};

export const DeployButton = () => {
  return (
    <Link
      target="_blank"
      href={`https://github.com/enskiilda/ai-python-computer-use`}
      className="flex flex-row gap-2 items-center bg-zinc-900 px-3 py-2 rounded-md text-zinc-50 hover:bg-zinc-950 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-50"
    >
      <VercelIcon size={14} />
      Source
    </Link>
  );
};
