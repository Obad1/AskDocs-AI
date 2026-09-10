import React from "react";
import Drawer from "./Drawer";
import { SummaryGranularitySlider } from "../assistant/SummaryGranularitySlider";

/**
 * Summary generator drawer — the (context-free) summary tool relocated out of
 * the permanently-pinned right-pane header into a panel the user opens on
 * demand. Keeps the default canvas focused on the active document/chat.
 */
export default function SummaryDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  return (
    <Drawer open={open} title="Summarize documents" onClose={onClose}>
      <SummaryGranularitySlider />
    </Drawer>
  );
}