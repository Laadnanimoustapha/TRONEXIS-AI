# Design Guidelines: AI Text Processing Chat Application

## Design Approach: Design System + Reference-Based Hybrid

**Selected Approach**: Material Design principles with inspiration from modern AI chat interfaces (ChatGPT, Claude, Perplexity)
**Justification**: Utility-focused productivity tool requiring clear information hierarchy, efficient interaction patterns, and professional trustworthiness

## Core Design Elements

### A. Color Palette

**Dark Mode (Primary)**
- Background Primary: 222 14% 10% (Deep charcoal)
- Background Secondary: 222 14% 14% (Chat message container)
- Background Elevated: 222 14% 18% (Input area, cards)
- Text Primary: 0 0% 98% (High contrast)
- Text Secondary: 220 9% 65% (Muted text)
- Accent Primary: 217 91% 60% (Vibrant blue for actions)
- Accent Hover: 217 91% 55%
- Success: 142 76% 45% (Summarize mode indicator)
- Warning: 38 92% 50% (Reformulate mode indicator)
- Border Subtle: 222 14% 25%

**Light Mode**
- Background Primary: 0 0% 100%
- Background Secondary: 220 14% 96%
- Background Elevated: 0 0% 100%
- Text Primary: 222 47% 11%
- Text Secondary: 215 14% 34%

### B. Typography

**Font Stack**: 'Inter' from Google Fonts
- Headings: 600 weight, tracking-tight
- Body: 400 weight, leading-relaxed
- Code/Monospace: 'JetBrains Mono' for any technical text
- Sizes: text-sm (14px), text-base (16px), text-lg (18px), text-xl (20px)

### C. Layout System

**Spacing Primitives**: Use Tailwind units of **2, 3, 4, 6, 8, 12** consistently
- Component padding: p-4 or p-6
- Section spacing: space-y-4, space-y-6
- Container max-width: max-w-4xl for chat area
- Message bubbles: px-4 py-3 or px-6 py-4

**Grid Structure**:
- Single-column centered layout for chat (max-w-4xl mx-auto)
- Full-width header/footer with contained content
- Sticky input area at bottom

### D. Component Library

**Navigation/Header**
- Minimal top bar with app logo/name
- Mode toggle (Summarize/Reformulate) as prominent pill-style segmented control
- Dark/light mode toggle (optional utility)
- Height: h-16, sticky positioning

**Chat Interface**
- Messages container: flex flex-col space-y-4, p-6
- User messages: Right-aligned, accent blue background, rounded-2xl rounded-br-md
- AI responses: Left-aligned, secondary background, rounded-2xl rounded-bl-md
- Max message width: max-w-[85%]
- Typography: leading-relaxed for readability

**Input Area**
- Sticky bottom: sticky bottom-0 with backdrop-blur
- Textarea: Rounded-2xl border, focus:ring-2 ring-accent
- Auto-resize on text input
- Send button: Icon button with accent color, absolute positioned inside textarea
- Character counter (subtle, text-secondary)

**Mode Toggle**
- Segmented control design with smooth transition
- Active state: Filled with mode-specific color (success for summarize, warning for reformulate)
- Inactive state: Transparent with border
- Icons + labels for clarity
- Width: inline-flex, center-aligned

**Loading States**
- Typing indicator: Three animated dots in AI message bubble
- Shimmer effect for pending responses
- Subtle pulse animation

**Empty State**
- Centered welcome message
- Quick action cards showing example prompts
- Mode-specific suggestions (2x2 grid on desktop, stacked on mobile)

### E. Interaction Patterns

**Animations**: Minimal and purposeful only
- Message appearance: Slide-up fade-in (150ms)
- Mode switch: Color transition (200ms)
- Button hover: Subtle scale (1.02) and brightness
- Focus states: 2px ring with accent color

**Micro-interactions**:
- Send button disabled state when textarea empty
- Enter to send, Shift+Enter for new line
- Copy button on hover for AI responses
- Scroll to bottom on new message

### F. Responsive Behavior

- Mobile (< 768px): px-4 spacing, full-width messages
- Tablet (768px - 1024px): px-6 spacing, max-w-3xl
- Desktop (> 1024px): px-8 spacing, max-w-4xl
- Input area adapts: Fixed height on mobile, auto-resize on desktop

### G. Accessibility Considerations

- Maintain WCAG AA contrast ratios (4.5:1 for text)
- Keyboard navigation for all interactive elements
- Focus indicators on all inputs and buttons
- ARIA labels for icon buttons
- Screen reader announcements for new AI messages

## Images

**No hero image required** - This is a utility-focused chat application where immediate functionality takes precedence. The interface should be clean and distraction-free, allowing users to start interacting immediately without scrolling past hero sections.

Optional decorative elements:
- Subtle gradient overlay in empty state background
- Abstract geometric patterns in mode toggle backgrounds (very subtle)
- App icon/logo in header (SVG, simple geometric design)