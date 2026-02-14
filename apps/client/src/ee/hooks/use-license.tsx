import { useAtom } from "jotai";
import { currentUserAtom } from "@/features/user/atoms/current-user-atom.ts";

export const useLicense = () => {
  const [currentUser] = useAtom(currentUserAtom);
  // ORIGINAL: return { hasLicenseKey: currentUser?.workspace?.hasLicenseKey };
  // BYPASS: Always return true to unlock all enterprise features
  return { hasLicenseKey: true };
};

export default useLicense;
